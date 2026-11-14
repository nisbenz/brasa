"""Step 2 - tensor creation: densify features.parquet into a memory-mapped stack (see lib_stack.py).

Usage: python -I ml/run_tensor.py <derived_dir> [--years 2012-2025] [--preset v2|legacy|all]
           [--bbox W,S,E,N] [--name overlap] [--features <parquet>] [--aux <parquet> ...]
Writes <derived>/ml/<name>/{X,Y,N20,S,CL}.npy + meta.json.

Presets (lib_config.PRESETS): "legacy" = the 15 combined m_* channels of the original U-Net,
"v2" = 30 per-satellite t_* / a_* channels, "all" = both (needed to score old and new models on
one stack). Size: 1.47 MB per channel-month on the default grid, e.g. 2012-2025 v2 = 7.4 GB,
2018-2025 all = 6.4 GB.

Transforms: count channels -> log1p; mean channels -> (x - mean) / sd over every non-null value in
the table (all years), 0 where undefined. Climatologies use MODIS only (complete years in the table),
never VIIRS, so they exist before 2012 and leak nothing about the target.

--aux: optional external covariates already on the 0.25-degree grid (e.g. MCD64A1 burned area or
MOD35 cloud fraction regridded per cell-month). Parquet with `cell`, optionally `year` + `month`,
and any value columns; each value column becomes a standardized channel "aux_<column>" (monthly
files -> X, static files -> S). Every covariate must exist for 2000 onward to be usable pre-VIIRS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import (AQUA_FIRST_MONTH, DEFAULT_COLS, DEFAULT_ROWS, GRID_COLS, PRESETS, SNAP,  # noqa: E402
                        STATIC_NAMES, channel_kind, excluded_months, parse_years)


def grid_window(bbox: str | None) -> tuple[int, int, int, int]:
    if not bbox:
        return (*DEFAULT_ROWS, *DEFAULT_COLS)
    w, s, e, n = map(float, bbox.split(","))
    return (int(np.floor((s + 90) * 4 / SNAP)) * SNAP, int(np.ceil((n + 90) * 4 / SNAP)) * SNAP,
            int(np.floor((w + 180) * 4 / SNAP)) * SNAP, int(np.ceil((e + 180) * 4 / SNAP)) * SNAP)


def transform(v: np.ndarray, kind: str, stats: tuple[float, float] | None) -> np.ndarray:
    if kind == "count":
        return np.log1p(np.nan_to_num(v))
    assert stats is not None
    mu, sd = stats
    return np.where(np.isfinite(v), (v - mu) / sd, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("derived", type=Path)
    ap.add_argument("--years", default="2012-2025")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v2")
    ap.add_argument("--bbox", default=None, help="W,S,E,N degrees, snapped outward to 2 degrees")
    ap.add_argument("--name", default="overlap")
    ap.add_argument("--features", type=Path, default=None, help="default <derived>/ml/features.parquet")
    ap.add_argument("--aux", type=Path, action="append", default=[])
    a = ap.parse_args()

    feat_path: Path = a.features or a.derived / "ml" / "features.parquet"
    feat = f"'{feat_path}'"
    out = a.derived / "ml" / a.name
    out.mkdir(parents=True, exist_ok=True)
    r0, r1, c0, c1 = grid_window(a.bbox)
    H, W = r1 - r0, c1 - c0
    in_grid = (f"cell // {GRID_COLS} >= {r0} AND cell // {GRID_COLS} < {r1} AND "
               f"cell % {GRID_COLS} >= {c0} AND cell % {GRID_COLS} < {c1}")
    con = duckdb.connect()
    cols_in_table = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {feat}").fetchall()}
    channels = list(PRESETS[a.preset])
    missing = [c for c in channels if c not in cols_in_table]
    if missing:
        raise SystemExit(f"{feat_path} lacks {missing[:4]}...: rebuild it with run_clean.py")

    # optional external covariates
    aux_month: list[tuple[str, str, str]] = []   # (channel name, file, column)
    aux_static: list[tuple[str, str, str]] = []
    for p in a.aux:
        cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM '{p}'").fetchall()]
        monthly = "year" in cols and "month" in cols
        for c in cols:
            if c not in ("cell", "year", "month"):
                (aux_month if monthly else aux_static).append((f"aux_{c}", str(p), c))
    channels += [n for n, _, _ in aux_month]

    yrs = set(parse_years(a.years))
    have = con.sql(f"SELECT DISTINCT year, month FROM {feat} ORDER BY ALL").fetchall()
    months = [(y, m) for y, m in have if y in yrs]
    T = len(months)
    tindex = {ym: i for i, ym in enumerate(months)}
    print(f"grid rows {r0}:{r1} cols {c0}:{c1} ({H}x{W}), {T} months, {len(channels)} channels "
          f"({T * len(channels) * H * W * 2 / 1e9:.1f} GB X)", flush=True)

    stats: dict[str, tuple[float, float]] = {}
    for c in channels:
        if channel_kind(c) != "mean" or c.startswith("aux_"):
            continue
        mu, sd = con.execute(f"SELECT avg({c}), stddev({c}) FROM {feat} WHERE {c} IS NOT NULL "
                             f"AND {'m_n > 0' if c.startswith('m_') else 'TRUE'} AND {in_grid}").fetchall()[0]
        stats[c] = (float(mu), float(sd or 1.0))
    for name, f, c in aux_month + aux_static:
        mu, sd = con.execute(f"SELECT avg({c}), stddev({c}) FROM '{f}' WHERE {c} IS NOT NULL").fetchall()[0]
        stats[name] = (float(mu), float(sd or 1.0))

    X = np.lib.format.open_memmap(out / "X.npy", "w+", np.float16, (T, len(channels), H, W))
    Y = np.lib.format.open_memmap(out / "Y.npy", "w+", np.float32, (T, H, W))
    N20 = np.lib.format.open_memmap(out / "N20.npy", "w+", np.float32, (T, H, W))
    for i, (y, _) in enumerate(months):
        Y[i] = np.nan if y < 2012 else 0
        N20[i] = np.nan if y < 2018 else 0
    base_cols = [c for c in channels if not c.startswith("aux_")]
    for y in sorted({y for y, _ in months}):
        df = con.sql(f"SELECT cell, month, v_n, n20_n, {', '.join(base_cols)} FROM {feat} "
                     f"WHERE year = {y} AND {in_grid}").df()
        t = np.array([tindex.get((y, m), -1) for m in df.month])
        df, t = df[t >= 0], t[t >= 0]
        r = df.cell.to_numpy() // GRID_COLS - r0
        c = df.cell.to_numpy() % GRID_COLS - c0
        for k, ch in enumerate(base_cols):
            v = df[ch].to_numpy(dtype=np.float64, na_value=np.nan)
            X[t, k, r, c] = transform(v, channel_kind(ch), stats.get(ch)).astype(np.float16)
        if y >= 2012:
            Y[t, r, c] = df.v_n.fillna(0).to_numpy(np.float32)
        if y >= 2018:
            N20[t, r, c] = df.n20_n.fillna(0).to_numpy(np.float32)
        for name, f, col in aux_month:
            k = channels.index(name)
            ax = con.sql(f"SELECT cell, month, {col} AS v FROM '{f}' WHERE year = {y} AND {in_grid}").df()
            ta = np.array([tindex.get((y, m), -1) for m in ax.month])
            ax, ta = ax[ta >= 0], ta[ta >= 0]
            X[ta, k, ax.cell.to_numpy() // GRID_COLS - r0, ax.cell.to_numpy() % GRID_COLS - c0] = \
                transform(ax.v.to_numpy(np.float64, na_value=np.nan), "mean", stats[name]).astype(np.float16)
        print(y, f"{len(df):,} rows", flush=True)
    X.flush(); Y.flush(); N20.flush()

    # MODIS-only climatology over complete years in the features table
    fy = [y for (y,) in con.sql(f"SELECT year FROM {feat} GROUP BY year "
                                f"HAVING count(DISTINCT month) = 12 ORDER BY year").fetchall()]
    cl = con.sql(f"""SELECT cell, month, coalesce(sum(m_n), 0)::DOUBLE / {len(fy)} AS mean_n FROM {feat}
                     WHERE year IN ({",".join(map(str, fy))}) AND {in_grid} GROUP BY ALL""").df()
    CL = np.zeros((12, H, W), np.float32)
    CL[cl.month.to_numpy() - 1, cl.cell.to_numpy() // GRID_COLS - r0,
       cl.cell.to_numpy() % GRID_COLS - c0] = np.log1p(cl.mean_n.to_numpy())
    lat = (np.arange(r0, r1) + 0.5) / 4 - 90
    lon = np.radians((np.arange(c0, c1) + 0.5) / 4 - 180)
    static = list(STATIC_NAMES) + [n for n, _, _ in aux_static]
    S = np.zeros((len(static), H, W), np.float32)
    S[0] = (lat / 90)[:, None]
    S[1] = np.sin(lon)[None, :]
    S[2] = np.cos(lon)[None, :]
    S[3] = np.log1p(np.expm1(CL).mean(0))
    for k, (name, f, col) in enumerate(aux_static, start=len(STATIC_NAMES)):
        ax = con.sql(f"SELECT cell, {col} AS v FROM '{f}' WHERE {in_grid}").df()
        S[k, ax.cell.to_numpy() // GRID_COLS - r0, ax.cell.to_numpy() % GRID_COLS - c0] = \
            transform(ax.v.to_numpy(np.float64, na_value=np.nan), "mean", stats[name])
    np.save(out / "CL.npy", CL)
    np.save(out / "S.npy", S)

    if "a_n_all" in cols_in_table:
        aq = con.sql(f"SELECT year, month FROM {feat} GROUP BY ALL HAVING sum(a_n_all) > 0").fetchall()
    else:
        aq = [ym for ym in have if ym >= AQUA_FIRST_MONTH]
    json.dump({
        "r0": r0, "r1": r1, "c0": c0, "c1": c1,
        "months": [f"{y}-{m:02d}" for y, m in months],
        "channels": channels,
        "static": static,
        "preset": a.preset,
        "norm": stats,
        "clim_years": fy,
        "aqua_months": sorted(f"{y}-{m:02d}" for y, m in aq),
        "excluded_months": excluded_months(),
        "features": str(feat_path),
    }, open(out / "meta.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

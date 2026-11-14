"""Step 4 - evaluation: compare any number of model checkpoints (old and new) with Brasa M3 and the
NOAA-20 ceiling on held-out months, and write a report.

Usage: python -I ml/run_evaluate.py <derived_dir> <stack_dir> --model old=model/unet.pt --model new=model/unet_v2.pt
           [--test 2019-2024] [--brasa-train 2012-2018] [--out <dir>] [--importance] [--terra-only]
The stack must contain every channel the models read (run_tensor.py --preset all serves both the
legacy and the v2 model) and the test months +-1.

Writes to --out (default <derived>/ml/eval):
  report.md          everything below as readable tables, plus warnings
  scores.csv         pipeline/03_validate.py metrics per model x scale (+ gap closed, anomaly slope)
  years.csv          5-degree error and bias per test year (sensor drift)
  bootstrap.csv      paired month-bootstrap 95% intervals of error differences, every scale
  regimes.csv        cell-month bias / error by fire regime, and where MODIS saw nothing
  bands.csv          bias by latitude band (coarse biome proxy)
  countries.csv      median monthly error in the 15 biggest fire countries
  calibration.json   randomized-PIT histogram and 80% interval coverage of each network
  importance.json    (--importance) permutation importance of input groups, per network
  summary.json       machine-readable headline numbers

Backcast check (the real use is 2000-2011, i.e. backwards in time): train a model on 2016-2025
and pass --test 2012-2014. Models that saw a test year in training are flagged in the report.
--terra-only also scores every per-satellite model with Aqua removed (simulated 2000-2002 months)
against a Terra-only Brasa refit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import GRID_COLS, parse_years  # noqa: E402
from lib_metrics import (BrasaMaps, Regions, RegionSeries, bootstrap_diff, gap_closed,  # noqa: E402
                         load_brasa_train, pit_summary, randomized_pit, score)
from lib_model import MonthPrediction, Predictor, pick_device  # noqa: E402
from lib_stack import Stack  # noqa: E402

BRASA, CEIL = "Brasa M3", "ceiling NOAA-20"
SCALES = ("1deg", "5deg", "country")
REGIME_EDGES = [1e-9, 1, 10, 100]
REGIME_LABELS = ["0", "<1", "1-10", "10-100", ">=100"]
BANDS = {"S extratropics (<23.5S)": (-90, -23.5), "tropics (23.5S-23.5N)": (-23.5, 23.5),
         "N subtropics / temperate (23.5-50N)": (23.5, 50), "boreal (>50N)": (50, 90)}
IMPORTANCE_GROUPS = {
    "counts": ("n", "n_all", "terra"), "FRP": ("frp_sum", "frp_max", "f10", "f30", "f100", "fbig"),
    "night share": ("night",), "fire days / burstiness": ("days", "maxday"),
    "pixel size / confidence / brightness": ("pix", "conf", "bright"), "local overpass time": ("hour",),
}


class CellStats:
    """Cell-level accumulators of one source: totals by regime / band / MODIS-zero, paired errors."""

    def __init__(self) -> None:
        self.regime = np.zeros(len(REGIME_LABELS))
        self.band = np.zeros(len(BANDS))
        self.none = 0.0
        self.err: dict[int, list[npt.NDArray[np.float64]]] = defaultdict(list)
        self.err_none: list[npt.NDArray[np.float64]] = []

    def add(self, p: np.ndarray, o: np.ndarray, M: np.ndarray, regime: np.ndarray, band: np.ndarray) -> None:
        p, o = np.nan_to_num(p.reshape(-1).astype(np.float64)), np.nan_to_num(o.reshape(-1))
        self.regime += np.bincount(regime, p, len(REGIME_LABELS))
        self.band += np.bincount(band, p, len(BANDS))
        zero = M.reshape(-1) == 0
        self.none += p[zero].sum()
        big = (p + o) >= 20
        e = np.abs(np.log((p[big] + 1) / (o[big] + 1)))
        rb = regime[big]
        for i in range(len(REGIME_LABELS)):
            self.err[i].append(e[rb == i])
        self.err_none.append(e[zero[big]])


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Markdown table without the tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("" if isinstance(v, float) and not np.isfinite(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("derived", type=Path)
    ap.add_argument("stack")
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH.pt (repeatable)")
    ap.add_argument("--test", default="2019-2024")
    ap.add_argument("--brasa-train", default="2012-2018")
    ap.add_argument("--features", type=Path, default=None, help="default <derived>/ml/features.parquet")
    ap.add_argument("--countries", type=Path, default=None, help="default <derived>/app/cell_country.parquet")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--pit-cells", type=int, default=200_000, help="cells sampled for calibration")
    ap.add_argument("--pit-draws", type=int, default=200)
    ap.add_argument("--terra-only", action="store_true")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--imp-months", type=int, default=12)
    a = ap.parse_args()
    t_start = time.time()
    dev = pick_device()
    out: Path = a.out or a.derived / "ml" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    features: Path = a.features or a.derived / "ml" / "features.parquet"
    countries: Path = a.countries or next(p for p in (a.derived / "app" / "cell_country.parquet",
                                                      a.derived / "cell_country.parquet") if p.exists())
    s = Stack(a.stack)
    H, W = s.shape
    warnings: list[str] = []

    # --- models ---------------------------------------------------------------------------------
    models: dict[str, Predictor] = {}
    for arg in a.model:
        name, _, path = arg.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=PATH, got {arg!r}")
        p = Predictor.load(path, dev, name)
        s.check(p.config.spec)
        models[name] = p
    test_years = parse_years(a.test)
    for name, p in models.items():
        seen = sorted(set(p.train_years) & set(test_years))
        if seen:
            warnings.append(f"**{name}** saw VIIRS for test year(s) {seen} during training or early "
                            f"stopping: its scores on those years are in-sample.")
    test_idx = s.usable(test_years)
    if not test_idx:
        raise SystemExit(f"no usable test months for {a.test} in {s.path}")
    for name, p in models.items():
        lacking = [s.months[t] for t in test_idx for dt in p.config.spec.offsets if s.neighbour(t, dt) is None]
        if lacking:
            warnings.append(f"{name}: {len(lacking)} neighbour months missing from the stack (zeros used); "
                            "build the stack one month wider than the test period.")
    test_months = [s.months[t] for t in test_idx]
    cal = np.array([m for _, m in test_months])
    years_of = np.array([y for y, _ in test_months])
    variants: list[tuple[str, Predictor, bool]] = [(n, p, False) for n, p in models.items()]
    if a.terra_only:
        variants += [(f"{n} [Terra-only]", p, True) for n, p in models.items() if p.config.spec.aqua_flag]
    print(f"device {dev} | test {len(test_idx)} months | models: " +
          ", ".join(f"{n} ({len(p.members)} member(s), {'legacy' if p.is_legacy else 'v2'})"
                    for n, p in models.items()), flush=True)

    # --- references -----------------------------------------------------------------------------
    regions = Regions.build(s.cells(), countries)
    win = (s.meta["r0"], s.meta["r1"], s.meta["c0"], s.meta["c1"])
    brasa_years = parse_years(a.brasa_train)
    tr = load_brasa_train(features, brasa_years, s.excluded, win)
    brasa = BrasaMaps.fit(tr, s.cells())
    clim = tr.groupby("cell").M.sum() / len(brasa_years)          # MODIS dets / yr: fire regime
    rc = np.zeros(H * W)
    flat = s.cells().reshape(-1)
    pos = pd.Series(np.arange(flat.size), index=flat)
    hit = clim.index.intersection(pos.index)
    rc[pos.loc[hit].to_numpy()] = clim.loc[hit].to_numpy()
    regime = np.digitize(rc, REGIME_EDGES).astype(np.int64)
    del tr
    brasa_t = None
    if a.terra_only and any(v[2] for v in variants):
        brasa_t = BrasaMaps.fit(load_brasa_train(features, brasa_years, s.excluded, win, "t_n"), s.cells())
        variants_ref = [f"{BRASA} [Terra-only]"]
    else:
        variants_ref = []
    lat = ((flat // GRID_COLS) + 0.5) / 4 - 90
    band = np.zeros(flat.size, np.int64)
    for i, (lo, hi) in enumerate(BANDS.values()):
        band[(lat >= lo) & (lat < hi)] = i

    # --- one pass over the test months ----------------------------------------------------------
    sources = [v[0] for v in variants] + [BRASA] + variants_ref
    series = {k: RegionSeries(regions) for k in sources}
    obs = RegionSeries(regions)
    ceil_series, ceil_obs, ceil_rows = RegionSeries(regions), RegionSeries(regions), []
    cells: dict[str, CellStats] = {k: CellStats() for k in sources + [CEIL]}
    obs_regime, obs_band, obs_none = np.zeros(len(REGIME_LABELS)), np.zeros(len(BANDS)), 0.0
    pit: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
    rng = np.random.default_rng(0)
    per_month_pit = max(a.pit_cells // len(test_idx), 1)
    for k, t in enumerate(test_idx):
        t0 = time.time()
        y, mo = s.months[t]
        o = s.target(t)
        M = s.modis_count(t)
        o_flat = np.nan_to_num(o.reshape(-1).astype(np.float64))
        obs.add(o)
        obs_regime += np.bincount(regime, o_flat, len(REGIME_LABELS))
        obs_band += np.bincount(band, o_flat, len(BANDS))
        obs_none += o_flat[M.reshape(-1) == 0].sum()
        cand = np.nonzero(np.isfinite(o.reshape(-1)))[0]
        for name, p, off in variants:
            pr: MonthPrediction = p.predict_month(s, t, dev, aqua_off=off)
            mean = pr.mean
            series[name].add(mean)
            cells[name].add(mean, o, M, regime, band)
            busy = cand[(o.reshape(-1)[cand] > 0) | (mean.reshape(-1)[cand] > 0.05)]
            rows = rng.choice(busy, min(per_month_pit, len(busy)), replace=False) if len(busy) else busy
            if len(rows):
                draws = pr.sample(a.pit_draws, rows, rng)
                pit[name].append(randomized_pit(draws, o.reshape(-1)[rows].astype(np.float32), rng))
        b = brasa.predict(M, mo)
        series[BRASA].add(b)
        cells[BRASA].add(b, o, M, regime, band)
        if brasa_t is not None:
            Mt = np.expm1(s.channel(t, "t_n"))
            bt = brasa_t.predict(Mt, mo)
            series[variants_ref[0]].add(bt)
            cells[variants_ref[0]].add(bt, o, M, regime, band)
        if y >= 2018:
            n20 = s.ceiling(t)
            ceil_series.add(n20)
            ceil_obs.add(o)
            ceil_rows.append(k)
            cells[CEIL].add(n20, o, M, regime, band)
        print(f"{y}-{mo:02d} ({k + 1}/{len(test_idx)}) {time.time() - t0:.1f}s", flush=True)

    # --- scores per scale -----------------------------------------------------------------------
    O = {sc: obs.array(sc) for sc in SCALES}
    P = {name: {sc: series[name].array(sc) for sc in SCALES} for name in sources}
    rows_score: list[dict[str, Any]] = []
    for sc in SCALES:
        for name in sources:
            rows_score.append({"model": name, "scale": sc, **score(P[name][sc], O[sc], cal)})
        if ceil_rows:
            cr = np.array(ceil_rows)
            rows_score.append({"model": CEIL, "scale": sc,
                               **score(ceil_series.array(sc), ceil_obs.array(sc), cal[cr])})
    T = pd.DataFrame(rows_score)
    for sc in SCALES:
        e = T[T.scale == sc].set_index("model").med_abs_log_err
        if BRASA in e and CEIL in e:
            T.loc[T.scale == sc, "gap_closed"] = [gap_closed(v, e[BRASA], e[CEIL]) for v in T[T.scale == sc].med_abs_log_err]
    T.to_csv(out / "scores.csv", index=False)

    # per year at 5 degrees
    yr_rows = []
    for yv in sorted(set(years_of)):
        sel = years_of == yv
        for name in sources:
            r = score(P[name]["5deg"][sel], O["5deg"][sel], cal[sel])
            yr_rows.append({"year": int(yv), "model": name, "med_abs_log_err": r["med_abs_log_err"],
                            "bias_pct": r["bias_pct"]})
    Yr = pd.DataFrame(yr_rows)
    Yr.to_csv(out / "years.csv", index=False)

    # paired bootstrap: every network vs Brasa, and every pair of networks (listed order)
    nets = [v[0] for v in variants]
    pairs = [(n, BRASA) for n in nets] + [(nets[j], nets[i]) for i in range(len(nets)) for j in range(i + 1, len(nets))]
    if variants_ref:
        pairs += [(n, variants_ref[0]) for n in nets if n.endswith("[Terra-only]")]
    B = pd.DataFrame([{"scale": sc, "a": x, "b": y_, **dict(zip(("diff", "lo", "hi"),
                       bootstrap_diff(P[x][sc], P[y_][sc], O[sc], a.bootstrap)))}
                      for sc in SCALES for x, y_ in pairs])
    B["verdict"] = np.where(B.hi < 0, "a better", np.where(B.lo > 0, "b better", "no clear difference"))
    B.to_csv(out / "bootstrap.csv", index=False)

    # regimes, MODIS-zero cells, latitude bands
    reg_rows = []
    for name in sources + ([CEIL] if ceil_rows else []):
        c = cells[name]
        for i, lab in enumerate(REGIME_LABELS):
            e = np.concatenate(c.err[i]) if c.err[i] else np.array([])
            reg_rows.append({"model": name, "regime": lab,
                             "bias_pct": round(100 * (c.regime[i] / obs_regime[i] - 1), 1) if obs_regime[i] else np.nan,
                             "cell_med_abs_log_err": round(float(np.median(e)), 3) if e.size else np.nan})
        e = np.concatenate(c.err_none) if c.err_none else np.array([])
        reg_rows.append({"model": name, "regime": "MODIS saw nothing",
                         "bias_pct": round(100 * (c.none / obs_none - 1), 1) if obs_none else np.nan,
                         "cell_med_abs_log_err": round(float(np.median(e)), 3) if e.size else np.nan})
    if ceil_rows:
        warnings.append("Ceiling rows in regimes.csv / bands.csv compare NOAA-20 against all test-month "
                        "S-NPP totals; they are only exact when every test year is >= 2018.")
    R = pd.DataFrame(reg_rows)
    R.to_csv(out / "regimes.csv", index=False)
    Bd = pd.DataFrame([{"model": name, "band": lab,
                        "bias_pct": round(100 * (cells[name].band[i] / obs_band[i] - 1), 1) if obs_band[i] else np.nan}
                       for name in sources for i, lab in enumerate(BANDS)])
    Bd.to_csv(out / "bands.csv", index=False)

    # biggest fire countries
    names = regions.names["country"]
    tot = O["country"].sum(0)
    top = [i for i in np.argsort(tot)[::-1] if names[i] != "?"][:15]
    Cn = pd.DataFrame({name: {names[i]: round(float(np.median(np.abs(np.log((P[name]["country"][:, i] + 1)
                                                                           / (O["country"][:, i] + 1))))), 3)
                              for i in top} for name in sources})
    Cn.index.name = "country"
    Cn.to_csv(out / "countries.csv")

    calib = {name: pit_summary(np.concatenate(v)) for name, v in pit.items() if v}
    json.dump(calib, open(out / "calibration.json", "w"), indent=1)

    # --- permutation importance -----------------------------------------------------------------
    imp: dict[str, dict[str, float]] = {}
    if a.importance:
        idx = list(rng.choice(test_idx, size=min(a.imp_months, len(test_idx)), replace=False))
        cal_i = np.array([s.months[t][1] for t in idx])
        obs_i = RegionSeries(regions)
        for t in idx:
            obs_i.add(s.target(t))

        def err(p: Predictor, **kw: Any) -> float:
            ser = RegionSeries(regions)
            for t in idx:
                ser.add(p.predict_month(s, t, dev, **{k: (v(t) if callable(v) else v) for k, v in kw.items()}).mean)
            return score(ser.array("5deg"), obs_i.array("5deg"), cal_i)["med_abs_log_err"]

        for name, p in models.items():
            base = err(p)
            chans = p.config.spec.channels
            other = {t: int(rng.choice([u for u in test_idx if u != t])) for t in idx}
            groups: dict[str, list[str]] = {g: [c for c in chans if c.split("_", 1)[1] in d]
                                            for g, d in IMPORTANCE_GROUPS.items()}
            if p.config.spec.aqua_flag:
                groups["Aqua block (all a_*)"] = [c for c in chans if c.startswith("a_")]
                groups["Terra block (all t_*)"] = [c for c in chans if c.startswith("t_")]
            imp[name] = {}
            for g, cs in groups.items():
                if cs:
                    imp[name][g] = round(err(p, swap=lambda t, cs=cs: {c: other[t] for c in cs}) - base, 4)
            imp[name]["location + climatology (shifted)"] = round(
                err(p, static_shift=(H // 3, W // 3)) - base, 4)
            imp[name] = dict(sorted(imp[name].items(), key=lambda kv: -kv[1]))
            print(f"importance {name}: {imp[name]}", flush=True)
        json.dump({"months": [s.meta["months"][t] for t in idx], "increase_5deg_err": imp},
                  open(out / "importance.json", "w"), indent=1)

    # --- report ---------------------------------------------------------------------------------
    head = T.pivot_table(index="model", columns="scale", values=["med_abs_log_err", "anomaly_sign_agree",
                                                                 "bias_pct", "gap_closed", "anomaly_slope"],
                         sort=False)
    order = sources + ([CEIL] if ceil_rows else [])
    lines = [f"# Model comparison, test {a.test} ({len(test_idx)} months)", "",
             f"Stack `{s.path}`, Brasa M3 refit on {a.brasa_train}. Models: " +
             ", ".join(f"`{n}` ({len(p.members)} member(s), trained on {p.train_years[0]}-{p.train_years[-1]})"
                       if p.train_years else f"`{n}`" for n, p in models.items()) + ".", ""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    for metric, title in (("med_abs_log_err", "Median |log error| of monthly region totals (lower is better)"),
                          ("anomaly_sign_agree", "Anomaly sign agreement (higher is better)"),
                          ("gap_closed", "Share of the Brasa -> NOAA-20 gap closed"),
                          ("bias_pct", "Bias % (sum predicted / sum observed - 1)"),
                          ("anomaly_slope", "Anomaly slope (1 = right year-to-year amplitude, < 1 = damped)")):
        if metric in head.columns.get_level_values(0):
            tb = head[metric].reindex(index=[o_ for o_ in order if o_ in head.index], columns=list(SCALES))
            lines += [f"## {title}", "", md_table(tb, "{:.3f}" if metric != "bias_pct" else "{:.2f}"), ""]
    lines += ["## Paired bootstrap (95% interval of a - b, median |log error|)", "",
              md_table(B.set_index("scale"), "{:.4f}"), ""]
    lines += ["## 5-degree error by year", "",
              md_table(Yr.pivot(index="year", columns="model", values="med_abs_log_err")[sources]), "",
              "## Bias % by year", "",
              md_table(Yr.pivot(index="year", columns="model", values="bias_pct")[sources], "{:.2f}"), ""]
    lines += ["## Fire regimes (MODIS detections / yr in the Brasa training years), cell-months", "",
              "Bias %:", "", md_table(R.pivot(index="regime", columns="model", values="bias_pct")
                                      .reindex(REGIME_LABELS + ["MODIS saw nothing"]), "{:.1f}"), "",
              "Cell median |log error| (pred + obs >= 20):", "",
              md_table(R.pivot(index="regime", columns="model", values="cell_med_abs_log_err")
                       .reindex(REGIME_LABELS + ["MODIS saw nothing"])), ""]
    lines += ["## Bias % by latitude band", "",
              md_table(Bd.pivot(index="band", columns="model", values="bias_pct")[sources], "{:.1f}"), ""]
    lines += ["## Top-15 fire countries: median monthly |log error|", "", md_table(Cn), ""]
    if calib:
        cdf = pd.DataFrame({k: {"coverage_80 (target 0.80)": v["coverage_80"], "PIT mean (target 0.50)": v["pit_mean"],
                                "PIT hist (10 bins, flat = 1)": " ".join(f"{x:.2f}" for x in v["pit_hist"])}
                            for k, v in calib.items()}).T
        cdf.index.name = "model"
        lines += ["## Calibration of the per-cell predictive distribution (randomized PIT)", "", md_table(cdf), ""]
    if imp:
        lines += ["## Permutation importance (increase of 5-degree error when a group is scrambled)", ""]
        for name, d in imp.items():
            lines += [f"**{name}**: " + ", ".join(f"{g} {v:+.3f}" for g, v in d.items()), ""]
    (out / "report.md").write_text("\n".join(lines))
    summary = {"test": a.test, "months": len(test_idx),
               "scores": T.to_dict(orient="records"), "bootstrap": B.to_dict(orient="records"),
               "warnings": warnings, "seconds": round(time.time() - t_start)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print("\n".join(lines[:60]))
    print(f"\nwrote {out} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()

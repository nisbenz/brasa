"""Densify ml/features.parquet into memory-mapped map stacks for the CNN.

Usage: python -I ml/build_tensor.py <derived_dir> [--years 2012-2025] [--bbox W,S,E,N] [--name overlap]
Writes <derived>/ml/<name>/
  X.npy    (T, C, H, W) float16  MODIS channels (log counts, log FRP, standardized means)
  Y.npy    (T, H, W)    float32  VIIRS S-NPP good detections (NaN before 2012)
  N20.npy  (T, H, W)    float32  VIIRS NOAA-20 good detections (NaN before 2018)
  S.npy    (4, H, W)    float32  static: lat/90, sin lon, cos lon, log1p MODIS annual climatology
  CL.npy   (12, H, W)   float32  log1p MODIS climatology per calendar month
  meta.json              grid origin, months, channel names, normalization, excluded months
Climatologies use MODIS only (all full years in features.parquet), never VIIRS, so they are
available for pre-VIIRS inference and leak nothing about the target.
Default grid: latitude -56..72 (rows 136..647), all longitudes; 512 x 1440 cells."""  """
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np

COUNT = [
    "m_n",
    "m_n_all",
    "m_terra",
    "m_night",
    "m_f10",
    "m_f30",
    "m_f100",
    "m_fbig",
    "m_days",
    "m_maxday",
]
FRP = ["m_frp_sum", "m_frp_max"]
MEANS = ["m_pix", "m_conf", "m_bright"]
CHANNELS = COUNT + FRP + MEANS
SNAP = 8  # U-Net downsamples 3x

ap = argparse.ArgumentParser()
ap.add_argument("derived", type=Path)
ap.add_argument("--years", default="2012-2025")
ap.add_argument(
    "--bbox", default=None, help="W,S,E,N degrees, snapped outward to 2 degrees"
)
ap.add_argument("--name", default="overlap")
args = ap.parse_args()

d = args.derived
feat = f"'{d}/ml/features.parquet'"
out = d / "ml" / args.name
out.mkdir(parents=True, exist_ok=True)

if args.bbox:
    w, s, e, n = map(float, args.bbox.split(","))
    r0 = int(np.floor((s + 90) * 4 / SNAP)) * SNAP
    r1 = int(np.ceil((n + 90) * 4 / SNAP)) * SNAP
    c0 = int(np.floor((w + 180) * 4 / SNAP)) * SNAP
    c1 = int(np.ceil((e + 180) * 4 / SNAP)) * SNAP
else:
    r0, r1, c0, c1 = 136, 648, 0, 1440
H, W = r1 - r0, c1 - c0

y0, y1 = map(int, args.years.split("-"))
con = duckdb.connect()
have = con.sql(f"SELECT DISTINCT year, month FROM {feat} ORDER BY ALL").fetchall()
months = [(y, m) for y, m in have if y0 <= y <= y1]
T = len(months)
tindex = {ym: i for i, ym in enumerate(months)}
print(
    f"grid rows {r0}:{r1} cols {c0}:{c1} ({H}x{W}), {T} months, {len(CHANNELS)} channels"
)

in_grid = f"cell // 1440 >= {r0} AND cell // 1440 < {r1} AND cell % 1440 >= {c0} AND cell % 1440 < {c1}"
# normalization stats for mean-type channels (inputs only, all years)
stats = {}
for c in MEANS:
    mu, sd = con.execute(
        f"SELECT avg({c}), stddev({c}) FROM {feat} WHERE m_n > 0 AND {in_grid}"
    ).fetchall()[0]
    stats[c] = (float(mu), float(sd or 1.0))

X = np.lib.format.open_memmap(out / "X.npy", "w+", np.float16, (T, len(CHANNELS), H, W))
Y = np.lib.format.open_memmap(out / "Y.npy", "w+", np.float32, (T, H, W))
N20 = np.lib.format.open_memmap(out / "N20.npy", "w+", np.float32, (T, H, W))
for i, (y, m) in enumerate(months):
    X[i] = 0
    Y[i] = np.nan if y < 2012 else 0
    N20[i] = np.nan if y < 2018 else 0
for y in sorted({y for y, _ in months}):
    df = con.sql(f"SELECT * FROM {feat} WHERE year = {y} AND {in_grid}").df()
    t = np.array([tindex.get((y, m), -1) for m in df.month])
    df, t = df[t >= 0], t[t >= 0]
    r = (df.cell.to_numpy() // 1440) - r0
    c = (df.cell.to_numpy() % 1440) - c0
    for k, ch in enumerate(CHANNELS):
        v = df[ch].to_numpy(dtype=np.float64, na_value=np.nan)
        if ch in MEANS:
            v = np.where(
                df.m_n.fillna(0).to_numpy() > 0, (v - stats[ch][0]) / stats[ch][1], 0.0
            )
        else:
            v = np.log1p(np.nan_to_num(v))
        X[t, k, r, c] = np.nan_to_num(v).astype(np.float16)
    if y >= 2012:
        Y[t, r, c] = df.v_n.fillna(0).to_numpy(np.float32)
    if y >= 2018:
        N20[t, r, c] = df.n20_n.fillna(0).to_numpy(np.float32)
    print(y, len(df), "rows", flush=True)
X.flush()
Y.flush()
N20.flush()

# MODIS-only climatology over complete years in the features table
full_years = con.sql(
    f"""SELECT year FROM {feat} GROUP BY year HAVING count(DISTINCT month) = 12"""
).fetchall()
fy = [y for (y,) in full_years]
cl = con.sql(f"""
  SELECT cell, month, coalesce(sum(m_n), 0)::DOUBLE / {len(fy)} AS mean_n FROM {feat}
  WHERE year IN ({",".join(map(str, fy))}) AND {in_grid} GROUP BY ALL""").df()
CL = np.zeros((12, H, W), np.float32)
CL[
    cl.month.to_numpy() - 1,
    cl.cell.to_numpy() // 1440 - r0,
    cl.cell.to_numpy() % 1440 - c0,
] = np.log1p(cl.mean_n.to_numpy())
lat = (np.arange(r0, r1) + 0.5) / 4 - 90
lon = np.radians((np.arange(c0, c1) + 0.5) / 4 - 180)
S = np.zeros((4, H, W), np.float32)
S[0] = (lat / 90)[:, None]
S[1] = np.sin(lon)[None, :]
S[2] = np.cos(lon)[None, :]
S[3] = np.log1p(np.expm1(CL).mean(0))
np.save(out / "CL.npy", CL)
np.save(out / "S.npy", S)

repo_model = (
    Path(__file__).resolve().parent.parent / "docs" / "validation" / "model.json"
)
excluded = (
    json.loads(repo_model.read_text())["excluded_training_months"]
    if repo_model.exists()
    else []
)
json.dump(
    {
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "months": [f"{y}-{m:02d}" for y, m in months],
        "channels": CHANNELS,
        "static": ["lat", "sin_lon", "cos_lon", "clim_annual"],
        "norm": stats,
        "clim_years": fy,
        "excluded_months": excluded,
    },
    open(out / "meta.json", "w"),
    indent=1,
)
print("wrote", out)

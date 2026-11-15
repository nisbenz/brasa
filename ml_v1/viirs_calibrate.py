"""Measure how MODIS detection attributes map onto VIIRS S-NPP ones, for ml_v1/to_viirs.py.

Usage: python -I ml_v1/viirs_calibrate.py <derived_dir> <firms_zip_dir> [--year 2020] [--out <json>]
Reads  <derived>/points/{modis,viirs-snpp}_<year>.parquet     (pipeline/01_to_parquet.py)
       <zip_dir>/viirs-snpp_<year>_all_countries.zip          (only for bright_ti5, dropped by 01)
Writes <derived>/ml/viirs_calibration.json (default)

All quantiles are over good detections (same filters as features.py), split by day/night:
  modis.<D|N>.<field>    MODIS quantiles of frp, bright, scan, conf     (u = rank of a template)
  viirs.<D|N>.<field>    S-NPP quantiles of frp, bright_ti4, scan, track (value at the same u)
  floor.<D|N>.<field>    S-NPP quantiles in cell-months where MODIS saw nothing (small fires)
  h_share / floor_h_share   share of 'h' among good S-NPP detections
  floor_day_share           day share of floor detections
  local_hour.<D|N>       quantiles of local solar acquisition hour (UTC + lon/15)
  ti5.<D|N>              bright_ti5 quantiles per bright_ti4 bin (edges in ti5_edges)
Quantile mapping is monotone, so a MODIS fire that is hot / large / off-nadir for MODIS becomes
one that is hot / large / off-nadir for VIIRS, with VIIRS's own value distribution.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

Q = [round(float(q), 4) for q in np.linspace(0, 1, 201)]
CELL = ("(LEAST(CAST(floor((lat+90)*4) AS INTEGER),719)*1440 + "
        "LEAST(CAST(floor((lon+180)*4) AS INTEGER),1439))")
MGOOD = "type = 0 AND TRY_CAST(confidence AS INTEGER) >= 30"
VGOOD = "type = 0 AND confidence <> 'l'"
TI4_EDGES = list(range(295, 370, 5))     # bins for bright_ti5 | bright_ti4 (I4 saturates ~367 K)


def quantiles(con, src, where, fields):
    sel = ", ".join(f"quantile_cont({e}, {Q}) AS {name}" for name, e in fields.items())
    row = con.execute(f"SELECT {sel} FROM {src} WHERE {where}").fetchone()
    return {name: [round(float(v), 3) for v in row[i]] for i, name in enumerate(fields)}


def ti5_given_ti4(zpath, frac, seed=0):
    """bright_ti5 quantiles per bright_ti4 bin from a sample of the raw FIRMS CSVs."""
    frames = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            df = pl.read_csv(io.BytesIO(z.read(name)), infer_schema=False,
                             columns=["bright_ti4", "bright_ti5", "daynight", "confidence", "type"])
            if df.height:
                frames.append(df.sample(fraction=frac, seed=seed) if df.height > 50 else df)
    df = (pl.concat(frames)
          .filter((pl.col("type") == "0") & (pl.col("confidence") != "l"))
          .with_columns(pl.col("bright_ti4").cast(pl.Float64), pl.col("bright_ti5").cast(pl.Float64)))
    out = {}
    for dn in ("D", "N"):
        d = df.filter(pl.col("daynight") == dn)
        ti4, ti5 = d["bright_ti4"].to_numpy(), d["bright_ti5"].to_numpy()
        b = np.digitize(ti4, TI4_EDGES)
        pooled = np.quantile(ti5, Q)
        out[dn] = [[round(float(v), 2) for v in (np.quantile(ti5[b == k], Q) if (b == k).sum() >= 200
                                                  else pooled)]
                   for k in range(len(TI4_EDGES) + 1)]
    return out, df.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("zips", type=Path)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--ti5-sample", type=float, default=0.1, help="fraction of raw rows for ti5")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = a.out or a.derived / "ml" / "viirs_calibration.json"
    pts = a.derived / "points"
    mod, vir = f"'{pts}/modis_{a.year}.parquet'", f"'{pts}/viirs-snpp_{a.year}.parquet'"

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory}'; SET threads={a.threads}; "
                f"SET temp_directory='{a.derived}/duck_tmp'")
    cal = {"year": a.year, "q": Q, "modis": {}, "viirs": {}, "floor": {}, "local_hour": {}}
    hour = "((time // 100) + (time % 100) / 60.0 + lon / 15.0 + 48) % 24"
    for dn in ("D", "N"):
        w = f"daynight = '{dn}'"
        cal["modis"][dn] = quantiles(con, mod, f"{MGOOD} AND {w}", {
            "frp": "frp", "bright": "bright", "scan": "scan", "conf": "TRY_CAST(confidence AS INTEGER)"})
        cal["viirs"][dn] = quantiles(con, vir, f"{VGOOD} AND {w}", {
            "frp": "frp", "bright_ti4": "bright", "scan": "scan", "track": "track"})
        cal["local_hour"][dn] = quantiles(con, vir, f"{VGOOD} AND {w}", {"h": hour})["h"]
        print("quantiles", dn, flush=True)

    con.execute(f"""CREATE TEMP TABLE mcells AS SELECT DISTINCT {CELL} AS cell, month(date) AS month
                    FROM {mod} WHERE {MGOOD}""")
    con.execute(f"""CREATE TEMP VIEW floor_v AS
                    SELECT v.* FROM (SELECT *, {CELL} AS cell, month(date) AS month FROM {vir}
                                     WHERE {VGOOD}) v
                    ANTI JOIN mcells USING (cell, month)""")
    tot, day = con.execute("SELECT count(*), count(*) FILTER (WHERE daynight = 'D') FROM floor_v").fetchone()
    cal["floor_day_share"] = round(day / tot, 4)
    for dn in ("D", "N"):
        cal["floor"][dn] = quantiles(con, "floor_v", f"daynight = '{dn}'", {
            "frp": "frp", "bright_ti4": "bright", "scan": "scan", "track": "track"})
    print("floor detections", tot, flush=True)

    hs = lambda src: dict(con.execute(
        f"SELECT daynight, avg((confidence = 'h')::INT) FROM {src} WHERE {VGOOD} GROUP BY 1").fetchall())
    cal["h_share"] = {k: round(v, 4) for k, v in hs(vir).items() if k in ("D", "N")}
    cal["floor_h_share"] = {k: round(v, 4) for k, v in hs("floor_v").items() if k in ("D", "N")}

    cal["ti5_edges"] = TI4_EDGES
    cal["ti5"], n = ti5_given_ti4(a.zips / f"viirs-snpp_{a.year}_all_countries.zip", a.ti5_sample)
    print("ti5 sample rows", n, flush=True)
    out.write_text(json.dumps(cal))
    print("wrote", out)


if __name__ == "__main__":
    main()

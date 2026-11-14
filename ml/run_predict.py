"""Step 5 - prediction: VIIRS S-NPP-equivalent counts per cell-month from MODIS alone, any months.

Usage: python -I ml/run_predict.py <stack_dir> <model.pt> <out_dir> [--start 2000-11] [--end 2025-12]
           [--countries <cell_country.parquet>] [--quantiles] [--app <features.parquet>]
The stack must cover the wanted months +-1 (run_tensor.py --years 2000-2025 --name full); VIIRS is
not needed. Per-satellite (v2) models handle the Terra-only months (before 2002-07) through their
Aqua flag; legacy models cannot, so their default start is 2002-07.

Writes to <out_dir>:
  cells_<year>.parquet   cell, year, month, pred (ensemble mean) [+ p10, p90 with --quantiles]
                         for every cell with pred > 1e-3
  monthly_totals.csv     region (GLOBAL + countries), year, month, modis, pred, viirs_obs
  app_nn.parquet         (--app) cell, year, month, pred >= 0.005 for cells that ever had a
                         detection: the input of pipeline/08_merge_unet.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import AQUA_FIRST_MONTH, month_str, parse_month  # noqa: E402
from lib_metrics import Regions  # noqa: E402
from lib_model import Predictor, pick_device  # noqa: E402
from lib_stack import Stack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack")
    ap.add_argument("model")
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", default=None, help="YYYY-MM (default: first month the model supports)")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--countries", type=Path, default=None)
    ap.add_argument("--quantiles", action="store_true", help="add p10 / p90 from 200 predictive draws")
    ap.add_argument("--app", type=Path, default=None, help="features.parquet: also write app_nn.parquet")
    a = ap.parse_args()
    dev = pick_device()
    s = Stack(a.stack)
    p = Predictor.load(a.model, dev)
    s.check(p.config.spec)
    start = parse_month(a.start) if a.start else ((2000, 1) if p.config.spec.aqua_flag else AQUA_FIRST_MONTH)
    if not p.config.spec.aqua_flag and start < AQUA_FIRST_MONTH:
        print(f"warning: {p.name} was trained on Terra + Aqua only; months before "
              f"{month_str(AQUA_FIRST_MONTH)} will be under-predicted", flush=True)
    months = [t for t, ym in enumerate(s.months) if start <= ym <= parse_month(a.end)]
    if not months:
        raise SystemExit(f"no months in range; stack has {s.meta['months'][0]}..{s.meta['months'][-1]}")
    regions = Regions.build(s.cells(), a.countries, ("country",)) if a.countries else None
    a.out.mkdir(parents=True, exist_ok=True)
    cells = s.cells().reshape(-1)
    rng = np.random.default_rng(0)
    t0, year_rows, totals = time.time(), [], []
    for k, t in enumerate(months):
        y, mo = s.months[t]
        pr = p.predict_month(s, t, dev)
        u = pr.mean.reshape(-1)
        keep = np.nonzero(u > 1e-3)[0]
        df = pd.DataFrame({"cell": cells[keep].astype(np.int32), "year": np.int16(y), "month": np.int8(mo),
                           "pred": u[keep].astype(np.float32)})
        if a.quantiles and len(keep):
            q = np.quantile(pr.sample(200, keep, rng), [0.1, 0.9], axis=0)
            df["p10"], df["p90"] = q[0].astype(np.float32), q[1].astype(np.float32)
        year_rows.append(df)
        M = s.modis_count(t).reshape(-1)
        V = s.target(t).reshape(-1)
        has_v = bool(np.isfinite(V).any())
        totals.append({"region": "GLOBAL", "year": y, "month": mo, "modis": float(M.sum()),
                       "pred": float(u.sum()), "viirs_obs": float(np.nansum(V)) if has_v else np.nan})
        if regions is not None:
            sm, sp, sv = (regions.sums(v)["country"] for v in (M, u, V))
            for i, name in enumerate(regions.names["country"]):
                totals.append({"region": name, "year": y, "month": mo, "modis": sm[i], "pred": sp[i],
                               "viirs_obs": sv[i] if has_v else np.nan})
        if k + 1 == len(months) or s.months[months[k + 1]][0] != y:
            pd.concat(year_rows, ignore_index=True).to_parquet(a.out / f"cells_{y}.parquet")
            year_rows = []
            print(f"{y} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(totals).to_csv(a.out / "monthly_totals.csv", index=False)
    if a.app:
        dst = a.out / "app_nn.parquet"
        duckdb.sql(f"""COPY (
          SELECT c.cell::INTEGER AS cell, c.year::SMALLINT AS year, c.month::TINYINT AS month, c.pred::FLOAT AS pred
          FROM read_parquet('{a.out}/cells_*.parquet') c SEMI JOIN (SELECT DISTINCT cell FROM '{a.app}') e USING (cell)
          WHERE c.pred >= 0.005 ORDER BY cell, year, month
        ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        print("wrote", dst)
    print(f"wrote {a.out}: {len(months)} months, {s.meta['months'][months[0]]}..{s.meta['months'][months[-1]]}")


if __name__ == "__main__":
    main()

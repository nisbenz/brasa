"""Load the U-Net + MLP ensemble into the app dataset.

Usage: python -I 08_merge_unet.py <derived_dir> [--nn <app_nn.parquet>] [--legacy Brazil]
Reads <derived>/ml/app_nn.parquet (the ensemble's per-cell output: cell, year, month,
pred = VIIRS S-NPP-equivalent detections predicted from MODIS alone, 2002-07 .. 2025-12).

What changes in <derived>/app:
  harm.parquet    MODIS-era months (MTA, 2002-07 .. 2012-01) take h = pred and src = 'NN' in every
                  cell outside the legacy countries; VIIRS-era months keep observed h and take hm = pred
                  (the "converted MODIS" line in the sensor-agreement chart).
  months.parquet  those months become src 'NN'.
  legacy_cells.parquet  cells that stay on Brasa M3 (ml/README.md: the ensemble is less accurate than
                  M3 in Brazil). The engine applies M3's small-fire floor only to these cells; the
                  ensemble already predicts small fires itself.
  nn.json         model card and hold-out scores for the app.
Terra-only months (MT, 2000-11 .. 2002-06) are not covered by the ensemble and stay on M3.
The first run keeps the M3 files as harm_m3.parquet and months_m3.parquet, and every run rebuilds
from those, so it can be rerun safely.
"""
import argparse, json, shutil
from pathlib import Path
import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("derived", type=Path)
ap.add_argument("--nn", type=Path, default=None)
ap.add_argument("--legacy", nargs="*", default=["Brazil"])
a = ap.parse_args()
A = a.derived / "app"
nn = a.nn or a.derived / "ml" / "app_nn.parquet"
if not nn.exists():
    raise SystemExit(f"{nn} not found: write it with ml/predict_ensemble.py --app (ml/README.md)")
for f in ("harm", "months"):
    if not (A / f"{f}_m3.parquet").exists():
        shutil.copy(A / f"{f}.parquet", A / f"{f}_m3.parquet")

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'; SET preserve_insertion_order = false")
con.execute(f"CREATE VIEW nn AS SELECT cell, year, month, pred FROM '{nn}'")
con.execute(f"CREATE VIEW h3 AS SELECT * FROM '{A}/harm_m3.parquet'")
con.execute(f"CREATE VIEW mo3 AS SELECT * FROM '{A}/months_m3.parquet'")
names = ", ".join("'" + n.replace("'", "''") + "'" for n in a.legacy) or "NULL"
con.execute(f"CREATE TABLE leg AS SELECT cell FROM '{A}/cell_country.parquet' WHERE country IN ({names})")
lo, hi = con.execute("SELECT min(year::INTEGER * 100 + month), max(year::INTEGER * 100 + month) FROM nn").fetchone()
print(f"ensemble covers {lo // 100}-{lo % 100:02d} .. {hi // 100}-{hi % 100:02d}; "
      f"{con.execute('SELECT count(*) FROM leg').fetchone()[0]:,} legacy cells ({', '.join(a.legacy)})")

# month-level provenance: harmonized Terra+Aqua months covered by the ensemble become 'NN'
con.execute(f"""CREATE TABLE mo AS SELECT * REPLACE (
  CASE WHEN src = 'MTA' AND year::INTEGER * 100 + month BETWEEN {lo} AND {hi} THEN 'NN' ELSE src END AS src) FROM mo3""")

# cell-months: M3 rows for legacy cells and uncovered months pass through; everything else joins the ensemble.
# MODIS-era predictions get the same missing-days correction (cov_factor) M3 applies to its input counts.
con.execute(f"""CREATE TABLE harm AS
WITH keep AS (
  SELECT h3.* FROM h3 JOIN mo USING (year, month)
  WHERE h3.cell IN (SELECT cell FROM leg) OR NOT (year::INTEGER * 100 + month BETWEEN {lo} AND {hi})),
other AS (
  SELECT * FROM h3 WHERE cell NOT IN (SELECT cell FROM leg) AND year::INTEGER * 100 + month BETWEEN {lo} AND {hi}),
nnx AS (SELECT * FROM nn WHERE cell NOT IN (SELECT cell FROM leg)),
j AS (
  SELECT coalesce(o.cell, n.cell) AS cell, coalesce(o.year, n.year) AS year, coalesce(o.month, n.month) AS month,
         o.h, o.m, o.v_snpp, o.v_n20, n.pred
  FROM other o FULL OUTER JOIN nnx n USING (cell, year, month))
SELECT cell::INTEGER AS cell, year::SMALLINT AS year, month::TINYINT AS month,
  (CASE WHEN mo.src = 'NN' THEN coalesce(pred, 0) * mo.cov_factor ELSE coalesce(h, 0) END)::FLOAT AS h,
  (coalesce(pred, 0) * CASE WHEN mo.src = 'NN' THEN mo.cov_factor ELSE 1 END)::FLOAT AS hm,
  mo.src, coalesce(m, 0)::INTEGER AS m, coalesce(v_snpp, 0)::INTEGER AS v_snpp, coalesce(v_n20, 0)::INTEGER AS v_n20
FROM j JOIN mo USING (year, month)
UNION ALL
SELECT * FROM keep""")
con.execute(f"COPY (SELECT * FROM harm ORDER BY cell, year, month) TO '{A}/harm.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"COPY (SELECT * FROM mo ORDER BY year, month) TO '{A}/months.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT * FROM leg ORDER BY cell) TO '{A}/legacy_cells.parquet' (FORMAT parquet)")

card = dict(
    model="U-Net + per-pixel MLP ensemble (mean of the two)",
    weights="v1: model/unet.pt and the MLP from ml/colab_run_model.ipynb (commit 4d18e46)",
    period=f"{lo // 100}-{lo % 100:02d} .. {hi // 100}-{hi % 100:02d}",
    legacy_countries=a.legacy,
    # v1 hold-out 2019-2024, median |log error| of monthly totals (ml/README.md and ml/compare.py at commit 4d18e46)
    holdout=[
        dict(method="Brasa ratio model (M3)", key="m3", e1=0.304, e5=0.246, ec=0.199, sign5=0.768),
        dict(method="Per-pixel MLP", key="mlp", e1=0.291, e5=0.245, ec=0.208, sign5=0.851),
        dict(method="U-Net", key="unet", e1=0.266, e5=0.211, ec=0.184, sign5=0.863),
        dict(method="U-Net + MLP ensemble", key="ens", e1=0.261, e5=0.203, ec=0.175, sign5=0.867),
        dict(method="Two VIIRS satellites, best possible", key="ceil", e1=0.163, e5=0.105, ec=0.087, sign5=0.925),
    ],
)
(A / "nn.json").write_text(json.dumps(card, indent=1))

print(con.execute("""SELECT src, count(*) AS rows, round(sum(h) / 1e6, 2) AS h_millions
                     FROM harm GROUP BY src ORDER BY src""").df().to_string(index=False))
print(con.execute("""SELECT year, round(sum(h) / 1e6, 2) AS harmonized_millions FROM harm
                     WHERE year BETWEEN 2002 AND 2013 GROUP BY year ORDER BY year""").df().to_string(index=False))
print("now rebuild the globe: python -I pipeline/07_globe.py", a.derived)

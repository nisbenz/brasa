"""Rich MODIS descriptors + VIIRS targets per 0.25-degree cell x month.

Usage: python -I ml/features.py <derived_dir> [--years 2000-2025] [--bbox W,S,E,N]
Reads  <derived>/points/*.parquet               (from pipeline/01_to_parquet.py)
Writes <derived>/ml/features.parquet            one row per cell-month with any detection

Same filters as pipeline/02_aggregate.py:
  MODIS good = type 0 and confidence >= 30;  VIIRS good = type 0 and confidence != 'l'
Columns (MODIS, good detections unless noted):
  m_n        detections                   m_n_all   type-0 detections incl. low confidence
  m_terra    Terra detections             m_night   night detections
  m_frp_sum  total FRP (MW)               m_frp_max largest single FRP
  m_f10 / m_f30 / m_f100 / m_fbig         detections with FRP <10, 10-30, 30-100, >=100 MW
  m_pix      mean pixel footprint scan*track (km^2; large = off-nadir, misses small fires)
  m_conf     mean confidence              m_bright  mean 4 um brightness temperature (K)
  m_days     distinct days with fire      m_maxday  most detections on one day (burstiness)
  v_n        VIIRS S-NPP good detections  (target)
  n20_n      VIIRS NOAA-20 good detections (second instrument: noise ceiling)
"""
import argparse, re
from pathlib import Path
import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("derived", type=Path)
ap.add_argument("--years", default="2000-2025")
ap.add_argument("--bbox", default=None, help="W,S,E,N in degrees (default: whole world)")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--memory", default="4GB")
args = ap.parse_args()

y0, y1 = map(int, args.years.split("-"))
d = args.derived
files = [p for p in sorted((d / "points").glob("*.parquet"))
         if y0 <= int(re.search(r"_(\d{4})\.parquet$", p.name).group(1)) <= y1]
if not files:
    raise SystemExit(f"no point files for {args.years} in {d/'points'}")
(d / "ml").mkdir(exist_ok=True)

where = "TRUE"
if args.bbox:
    w, s, e, n = map(float, args.bbox.split(","))
    where = f"lat >= {s} AND lat < {n} AND lon >= {w} AND lon < {e}"

con = duckdb.connect()
con.execute(f"SET memory_limit='{args.memory}'; SET threads={args.threads};")
con.execute(f"SET temp_directory='{d}/duck_tmp'")
src = "read_parquet([" + ",".join(f"'{p}'" for p in files) + "])"
CELL = ("(LEAST(CAST(floor((lat+90)*4) AS INTEGER),719)*1440 + "
        "LEAST(CAST(floor((lon+180)*4) AS INTEGER),1439))")
MGOOD = "type = 0 AND TRY_CAST(confidence AS INTEGER) >= 30"
VGOOD = "type = 0 AND confidence <> 'l'"

con.execute(f"""
COPY (
WITH modis_day AS (           -- day level first, so burstiness (m_maxday) is available
  SELECT {CELL} AS cell, date,
         count(*) FILTER (WHERE {MGOOD}) AS n,
         count(*) FILTER (WHERE type = 0) AS n_all,
         count(*) FILTER (WHERE {MGOOD} AND satellite = 'Terra') AS terra,
         count(*) FILTER (WHERE {MGOOD} AND daynight = 'N') AS night,
         sum(frp) FILTER (WHERE {MGOOD}) AS frp_sum,
         max(frp) FILTER (WHERE {MGOOD}) AS frp_max,
         count(*) FILTER (WHERE {MGOOD} AND frp < 10) AS f10,
         count(*) FILTER (WHERE {MGOOD} AND frp >= 10 AND frp < 30) AS f30,
         count(*) FILTER (WHERE {MGOOD} AND frp >= 30 AND frp < 100) AS f100,
         count(*) FILTER (WHERE {MGOOD} AND frp >= 100) AS fbig,
         sum(scan * track) FILTER (WHERE {MGOOD}) AS pix_sum,
         sum(TRY_CAST(confidence AS INTEGER)) FILTER (WHERE {MGOOD}) AS conf_sum,
         sum(bright) FILTER (WHERE {MGOOD}) AS bright_sum
  FROM {src} WHERE sensor = 'modis' AND {where}
  GROUP BY ALL
),
modis AS (
  SELECT cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
         sum(n)::INTEGER AS m_n, sum(n_all)::INTEGER AS m_n_all,
         sum(terra)::INTEGER AS m_terra, sum(night)::INTEGER AS m_night,
         sum(frp_sum)::FLOAT AS m_frp_sum, max(frp_max)::FLOAT AS m_frp_max,
         sum(f10)::INTEGER AS m_f10, sum(f30)::INTEGER AS m_f30,
         sum(f100)::INTEGER AS m_f100, sum(fbig)::INTEGER AS m_fbig,
         (sum(pix_sum) / nullif(sum(n), 0))::FLOAT AS m_pix,
         (sum(conf_sum) / nullif(sum(n), 0))::FLOAT AS m_conf,
         (sum(bright_sum) / nullif(sum(n), 0))::FLOAT AS m_bright,
         count(*) FILTER (WHERE n > 0)::SMALLINT AS m_days,
         max(n)::INTEGER AS m_maxday
  FROM modis_day GROUP BY ALL
),
viirs AS (
  SELECT {CELL} AS cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
         count(*) FILTER (WHERE sensor = 'viirs-snpp' AND {VGOOD})::INTEGER AS v_n,
         count(*) FILTER (WHERE sensor = 'viirs-jpss1' AND {VGOOD})::INTEGER AS n20_n
  FROM {src} WHERE sensor <> 'modis' AND {where}
  GROUP BY ALL
)
SELECT cell, year, month,
       m_n, m_n_all, m_terra, m_night, m_frp_sum, m_frp_max, m_f10, m_f30, m_f100, m_fbig,
       m_pix, m_conf, m_bright, m_days, m_maxday, v_n, n20_n
FROM modis FULL OUTER JOIN viirs USING (cell, year, month)
WHERE coalesce(m_n_all, 0) + coalesce(v_n, 0) + coalesce(n20_n, 0) > 0
ORDER BY year, month, cell
) TO '{d}/ml/features.parquet' (FORMAT parquet, COMPRESSION zstd)
""")
print(con.execute(f"""SELECT count(*) rows, min(year), max(year), sum(m_n), sum(v_n), sum(n20_n)
                      FROM '{d}/ml/features.parquet'""").fetchall())

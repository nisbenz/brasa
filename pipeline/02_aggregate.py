"""Aggregate hotspot points to 0.25-degree cell x month x sensor.

Usage: python -I 02_aggregate.py <derived_dir>
Reads  <derived>/points/*.parquet
Writes <derived>/cell_month.parquet   (sensor, cell, year, month, n_all, n, frp)
       <derived>/cell_country.parquet (cell, country)  majority country per cell
  n   = type 0 (presumed vegetation fire) and not low confidence
        (MODIS confidence >= 30, VIIRS confidence != 'l')
  cell = row*1440 + col, row = floor((lat+90)*4), col = floor((lon+180)*4)
"""
import sys
from pathlib import Path
import duckdb

d = Path(sys.argv[1])
con = duckdb.connect()
con.execute("SET memory_limit='6GB'; SET threads=12;")
con.execute(f"SET temp_directory='{d}/duck_tmp'")
pts = f"'{d}/points/*.parquet'"
CELL = ("(LEAST(CAST(floor((lat+90)*4) AS INTEGER),719)*1440 + "
        "LEAST(CAST(floor((lon+180)*4) AS INTEGER),1439))")
GOOD = ("type = 0 AND CASE WHEN sensor='modis' THEN TRY_CAST(confidence AS INTEGER) >= 30 "
        "ELSE confidence <> 'l' END")

con.execute(f"""
COPY (
  SELECT sensor, CASE WHEN sensor='modis' THEN satellite ELSE NULL END AS sat, {CELL} AS cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
         count(*)::INTEGER AS n_all,
         count(*) FILTER (WHERE {GOOD})::INTEGER AS n,
         sum(frp) FILTER (WHERE {GOOD})::FLOAT AS frp
  FROM read_parquet({pts})
  GROUP BY ALL
) TO '{d}/cell_month.parquet' (FORMAT parquet, COMPRESSION zstd)
""")
con.execute(f"""
COPY (
  SELECT cell, arg_max(country, k) AS country FROM (
    SELECT {CELL} AS cell, country, count(*) AS k FROM read_parquet({pts})
    WHERE sensor = 'viirs-snpp' OR sensor = 'modis' GROUP BY ALL)
  GROUP BY cell
) TO '{d}/cell_country.parquet' (FORMAT parquet)
""")
print(con.execute(f"""
  SELECT sensor, count(*) rows, sum(n_all) n_all, sum(n) n
  FROM '{d}/cell_month.parquet' GROUP BY sensor ORDER BY sensor""").fetchall())

"""Global 1-degree harmonized fire grid for the 3D globe.

Usage: python -I 07_globe.py <derived_dir>
Writes <derived>/app/globe.bin (little-endian):
  uint32 n, uint32 n_years, uint32 year0
  int16  lat_idx[n]  (row 0 = 90S..89S)      int16 lon_idx[n] (col 0 = 180W..179W)
  float32 clim[n*12]       mean VIIRS-equivalent hotspots per calendar month, 2001-2025
  float32 annual[n*n_years] VIIRS-equivalent hotspots per year
Values include the small-fire floor for MODIS-era months, like the calendar API.
"""
import struct, sys
from pathlib import Path
import duckdb, numpy as np

d = Path(sys.argv[1]); A = d / "app"
Y0, Y1 = 2001, 2025
con = duckdb.connect()
P1 = "((cell // 1440) // 4) * 360 + (cell % 1440) // 4"
df = con.sql(f"""
WITH h AS (
  SELECT {P1} AS p1, year, month, sum(h) AS h FROM '{A}/harm.parquet' GROUP BY ALL),
fa AS (
  SELECT {P1} AS p1, month, sum(b_ta) AS b_ta, sum(b_t) AS b_t FROM '{A}/floor.parquet' GROUP BY ALL),
fu AS (
  SELECT {P1} AS p1, year, month, sum(f.b_ta) AS b_ta, sum(f.b_t) AS b_t
  FROM '{A}/harm.parquet' x JOIN '{A}/floor.parquet' f USING (cell, month)
  WHERE x.src IN ('MT','MTA') AND x.m > 0 GROUP BY ALL),
keys AS (  -- every (p1, year, month) that has fire or a floor
  SELECT p1, year, month FROM h
  UNION SELECT fa.p1, mo.year, mo.month FROM fa JOIN '{A}/months.parquet' mo USING (month) WHERE mo.src IN ('MT','MTA'))
SELECT k.p1, k.year, k.month,
  coalesce(h.h,0) + CASE mo.src WHEN 'MT'  THEN greatest(coalesce(fa.b_t,0)  - coalesce(fu.b_t,0), 0)
                                WHEN 'MTA' THEN greatest(coalesce(fa.b_ta,0) - coalesce(fu.b_ta,0), 0) ELSE 0 END AS v
FROM keys k JOIN '{A}/months.parquet' mo USING (year, month)
LEFT JOIN h USING (p1, year, month) LEFT JOIN fa USING (p1, month) LEFT JOIN fu USING (p1, year, month)
WHERE k.year BETWEEN {Y0} AND {Y1}
""").df()

cells = np.sort(df.p1.unique())
idx = {c: i for i, c in enumerate(cells)}
n, ny = len(cells), Y1 - Y0 + 1
ii = df.p1.map(idx).to_numpy()
clim = np.zeros((n, 12), np.float64)
np.add.at(clim, (ii, df.month.to_numpy() - 1), df.v.to_numpy())
clim /= ny
annual = np.zeros((n, ny), np.float64)
np.add.at(annual, (ii, df.year.to_numpy() - Y0), df.v.to_numpy())
keep = annual.sum(1) >= 1  # drop cells with less than one hotspot in 25 years
cells, clim, annual = cells[keep], clim[keep], annual[keep]
n = len(cells)
lat, lon = (cells // 360).astype(np.int16), (cells % 360).astype(np.int16)
with open(A / "globe.bin", "wb") as f:
    f.write(struct.pack("<III", n, ny, Y0))
    f.write(lat.tobytes()); f.write(lon.tobytes())
    f.write(clim.astype("<f4").tobytes()); f.write(annual.astype("<f4").tobytes())
print(f"cells {n}, size {(A / 'globe.bin').stat().st_size / 1e6:.1f} MB, "
      f"global annual mean {annual.sum(0).mean() / 1e6:.2f} M, max cell-year {annual.max():.0f}")

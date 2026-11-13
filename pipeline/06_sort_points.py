"""Re-sort raw point files by latitude band then longitude, in small row groups,
so area queries (points CSV export) can skip row groups via min/max statistics.

Usage: python -I 06_sort_points.py <derived_dir>
Writes <derived>/points_sorted/, then swaps it in for <derived>/points (old kept as points_by_date/).
"""
import sys, time
from pathlib import Path
import duckdb

d = Path(sys.argv[1]); src = d / "points"; dst = d / "points_sorted"; dst.mkdir(exist_ok=True)
con = duckdb.connect(); con.execute("SET memory_limit='6GB'; SET threads=8; SET preserve_insertion_order=true;")
for f in sorted(src.glob("*.parquet")):
    out = dst / f.name
    if out.exists():
        continue
    t = time.time()
    con.execute(f"""COPY (SELECT * FROM '{f}' ORDER BY floor(lat), lon, date)
                    TO '{out}.tmp' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 50000)""")
    Path(f"{out}.tmp").rename(out)
    print(f"{f.name} {time.time() - t:.1f}s", flush=True)
n_src, n_dst = len(list(src.glob("*.parquet"))), len(list(dst.glob("*.parquet")))
if n_src == n_dst:
    src.rename(d / "points_by_date"); dst.rename(src)
    print("swapped: points/ is now latitude-sorted; old files in points_by_date/")

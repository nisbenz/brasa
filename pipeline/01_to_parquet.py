"""Convert FIRMS country-yearly zips to one Parquet file per sensor-year.

Usage: python -I 01_to_parquet.py <zip_dir> <out_dir> [workers]
Output: <out_dir>/points/<sensor>_<year>.parquet with a unified schema:
  lat, lon, date, time(hhmm int), satellite, confidence(str), frp, bright,
  scan, track, daynight, type, country, sensor
"""
import io
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import polars as pl

ZIP_RE = re.compile(r"^(modis|viirs-snpp|viirs-jpss1)_(\d{4})_all_countries\.zip$")


def convert(args):
    zpath, out = args
    m = ZIP_RE.match(zpath.name)
    sensor, year = m.group(1), m.group(2)
    dest = out / f"{sensor}_{year}.parquet"
    if dest.exists():
        return f"skip {dest.name}"
    frames = []
    prefix = f"{sensor}_{year}_"
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if not base.endswith(".csv"):
                continue
            country = base[len(prefix):-4] if base.startswith(prefix) else base[:-4]
            df = pl.read_csv(io.BytesIO(z.read(name)), infer_schema=False)
            if df.height == 0:
                continue
            bright = "brightness" if "brightness" in df.columns else "bright_ti4"
            frames.append(df.select(
                pl.col("latitude").cast(pl.Float32).alias("lat"),
                pl.col("longitude").cast(pl.Float32).alias("lon"),
                pl.col("acq_date").str.to_date().alias("date"),
                pl.col("acq_time").cast(pl.Int16).alias("time"),
                pl.col("satellite"),
                pl.col("confidence"),
                pl.col("frp").cast(pl.Float32),
                pl.col(bright).cast(pl.Float32).alias("bright"),
                pl.col("scan").cast(pl.Float32),
                pl.col("track").cast(pl.Float32),
                pl.col("daynight"),
                pl.col("type").cast(pl.Int8),
                pl.lit(country.replace("_", " ")).alias("country"),
            ))
    df = pl.concat(frames).with_columns(pl.lit(sensor).alias("sensor")).sort("date")
    tmp = dest.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd", row_group_size=500_000)
    tmp.rename(dest)
    return f"{dest.name}: {df.height:,} rows"


if __name__ == "__main__":
    zdir, out = Path(sys.argv[1]), Path(sys.argv[2]) / "points"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out.mkdir(parents=True, exist_ok=True)
    zips = sorted(p for p in zdir.iterdir() if ZIP_RE.match(p.name))
    # biggest first so the long ones don't straggle at the end
    zips.sort(key=lambda p: -p.stat().st_size)
    with Pool(workers) as pool:
        for msg in pool.imap_unordered(convert, [(z, out) for z in zips]):
            print(msg, flush=True)

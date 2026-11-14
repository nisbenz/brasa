"""Step 1 - data cleaning: FIRMS points -> one row per 0.25-degree cell x month.

Usage: python -I ml/run_clean.py <derived_dir> [--years 2000-2025] [--bbox W,S,E,N] [--out <parquet>]
Reads  <derived>/points/*.parquet          (from pipeline/01_to_parquet.py)
Writes <derived>/ml/features.parquet       (default; --out to write elsewhere)

Filters are those of pipeline/02_aggregate.py (lib_config.MODIS_GOOD / VIIRS_GOOD).
Columns, all from good detections unless noted:
  m_*  both MODIS satellites combined, exactly the columns the legacy U-Net was trained on:
       m_n, m_n_all (type 0 incl. low confidence), m_terra, m_night, m_frp_sum, m_frp_max,
       m_f10 / m_f30 / m_f100 / m_fbig (FRP < 10, 10-30, 30-100, >= 100 MW),
       m_pix (mean scan*track km^2; large = off-nadir), m_conf, m_bright (mean 4 um K),
       m_days (distinct fire days), m_maxday (most detections on one day)
  t_*, a_*  the same descriptors per satellite (Terra / Aqua) plus <sat>_hour: mean local solar
       time of daytime detections, which tracks the orbit drift of both satellites after ~2020.
       Aqua shares S-NPP's 13:30 overpass, Terra is at 10:30, so the split carries real signal.
  v_n  VIIRS S-NPP good detections (target);  n20_n  VIIRS NOAA-20 good detections (ceiling)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import GRID_COLS, MODIS_GOOD, SATELLITES, VIIRS_GOOD, parse_years

CELL: str = (f"(LEAST(CAST(floor((lat+90)*4) AS INTEGER),719)*{GRID_COLS} + "
             f"LEAST(CAST(floor((lon+180)*4) AS INTEGER),{GRID_COLS - 1}))")
LOCAL_HOUR: str = "((time // 100) + (time % 100) / 60.0 + lon / 15.0 + 48) % 24"
# per (cell, date, satellite) sums; every later aggregate is built from these
DAY_SUMS: dict[str, str] = {
    "n": f"count(*) FILTER (WHERE {MODIS_GOOD})",
    "n_all": "count(*) FILTER (WHERE type = 0)",
    "night": f"count(*) FILTER (WHERE {MODIS_GOOD} AND daynight = 'N')",
    "frp_sum": f"sum(frp) FILTER (WHERE {MODIS_GOOD})",
    "frp_max": f"max(frp) FILTER (WHERE {MODIS_GOOD})",
    "f10": f"count(*) FILTER (WHERE {MODIS_GOOD} AND frp < 10)",
    "f30": f"count(*) FILTER (WHERE {MODIS_GOOD} AND frp >= 10 AND frp < 30)",
    "f100": f"count(*) FILTER (WHERE {MODIS_GOOD} AND frp >= 30 AND frp < 100)",
    "fbig": f"count(*) FILTER (WHERE {MODIS_GOOD} AND frp >= 100)",
    "pix_sum": f"sum(scan * track) FILTER (WHERE {MODIS_GOOD})",
    "conf_sum": f"sum(TRY_CAST(confidence AS INTEGER)) FILTER (WHERE {MODIS_GOOD})",
    "bright_sum": f"sum(bright) FILTER (WHERE {MODIS_GOOD})",
    "hour_sum": f"sum({LOCAL_HOUR}) FILTER (WHERE {MODIS_GOOD} AND daynight = 'D')",
    "n_day": f"count(*) FILTER (WHERE {MODIS_GOOD} AND daynight = 'D')",
}
ADDITIVE: tuple[str, ...] = ("n", "n_all", "night", "frp_sum", "f10", "f30", "f100", "fbig",
                             "pix_sum", "conf_sum", "bright_sum", "hour_sum", "n_day")


def month_aggregates(prefix: str, src_n: str = "n") -> list[str]:
    """SELECT expressions for one satellite block (or the combined m_ block) from day-level rows."""
    p = prefix
    return [
        f"sum(n)::INTEGER AS {p}_n", f"sum(n_all)::INTEGER AS {p}_n_all",
        f"sum(night)::INTEGER AS {p}_night",
        f"sum(frp_sum)::FLOAT AS {p}_frp_sum", f"max(frp_max)::FLOAT AS {p}_frp_max",
        f"sum(f10)::INTEGER AS {p}_f10", f"sum(f30)::INTEGER AS {p}_f30",
        f"sum(f100)::INTEGER AS {p}_f100", f"sum(fbig)::INTEGER AS {p}_fbig",
        f"(sum(pix_sum) / nullif(sum(n), 0))::FLOAT AS {p}_pix",
        f"(sum(conf_sum) / nullif(sum(n), 0))::FLOAT AS {p}_conf",
        f"(sum(bright_sum) / nullif(sum(n), 0))::FLOAT AS {p}_bright",
        f"(sum(hour_sum) / nullif(sum(n_day), 0))::FLOAT AS {p}_hour",
        f"count(*) FILTER (WHERE {src_n} > 0)::SMALLINT AS {p}_days",
        f"max({src_n})::INTEGER AS {p}_maxday",
    ]


def build_sql(src: str, where: str, out: Path) -> str:
    day_cols = ",\n         ".join(f"{expr} AS {k}" for k, expr in DAY_SUMS.items())
    comb = ",\n         ".join(f"sum({k}) AS {k}" for k in ADDITIVE) + ",\n         max(frp_max) AS frp_max"
    # m_hour is not part of the legacy set; m_terra is filled from t_n in the final SELECT
    comb_month = [e for e in month_aggregates("m") if not e.endswith(" AS m_hour")]
    sat_blocks = []
    for p, name in SATELLITES.items():
        cols = ",\n           ".join(month_aggregates(p))
        sat_blocks.append(f"""{p}_month AS (
  SELECT cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
           {cols}
  FROM modis_sat_day WHERE satellite = '{name}' GROUP BY ALL
)""")
    legacy_cols = ["m_n", "m_n_all", "m_terra", "m_night", "m_frp_sum", "m_frp_max", "m_f10", "m_f30",
                   "m_f100", "m_fbig", "m_pix", "m_conf", "m_bright", "m_days", "m_maxday"]
    sat_cols = [f"{p}_{d}" for p in SATELLITES for d in
                ("n", "n_all", "night", "frp_sum", "frp_max", "f10", "f30", "f100", "fbig",
                 "pix", "conf", "bright", "hour", "days", "maxday")]
    joins = "\n".join(f"FULL OUTER JOIN {p}_month USING (cell, year, month)" for p in SATELLITES)
    return f"""
COPY (
WITH modis_sat_day AS (
  SELECT {CELL} AS cell, date, satellite,
         {day_cols}
  FROM {src} WHERE sensor = 'modis' AND {where}
  GROUP BY ALL
),
modis_day AS (                -- both satellites, day level, so burstiness (m_maxday) is defined
  SELECT cell, date,
         {comb}
  FROM modis_sat_day GROUP BY cell, date
),
m_month AS (
  SELECT cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
         {", ".join(comb_month)}
  FROM modis_day GROUP BY ALL
),
{",".join(sat_blocks)},
viirs AS (
  SELECT {CELL} AS cell, year(date)::SMALLINT AS year, month(date)::TINYINT AS month,
         count(*) FILTER (WHERE sensor = 'viirs-snpp' AND {VIIRS_GOOD})::INTEGER AS v_n,
         count(*) FILTER (WHERE sensor = 'viirs-jpss1' AND {VIIRS_GOOD})::INTEGER AS n20_n
  FROM {src} WHERE sensor <> 'modis' AND {where}
  GROUP BY ALL
)
SELECT cell, year, month, {", ".join(c for c in legacy_cols if c != "m_terra")},
       coalesce(t_n, CASE WHEN m_n IS NOT NULL THEN 0 END)::INTEGER AS m_terra,
       {", ".join(sat_cols)}, v_n, n20_n
FROM m_month {joins}
FULL OUTER JOIN viirs USING (cell, year, month)
WHERE coalesce(m_n_all, 0) + coalesce(v_n, 0) + coalesce(n20_n, 0) > 0
ORDER BY year, month, cell
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("derived", type=Path)
    ap.add_argument("--years", default="2000-2025")
    ap.add_argument("--bbox", default=None, help="W,S,E,N in degrees (default: whole world)")
    ap.add_argument("--out", type=Path, default=None, help="default <derived>/ml/features.parquet")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="4GB")
    a = ap.parse_args()

    yrs = set(parse_years(a.years))
    files = [p for p in sorted((a.derived / "points").glob("*.parquet"))
             if (m := re.search(r"_(\d{4})\.parquet$", p.name)) and int(m.group(1)) in yrs]
    if not files:
        raise SystemExit(f"no point files for {a.years} in {a.derived / 'points'}")
    out: Path = a.out or a.derived / "ml" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    where = "TRUE"
    if a.bbox:
        w, s, e, n = map(float, a.bbox.split(","))
        where = f"lat >= {s} AND lat < {n} AND lon >= {w} AND lon < {e}"

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory}'; SET threads={a.threads}; "
                f"SET temp_directory='{a.derived}/duck_tmp'; SET preserve_insertion_order=false;")
    src = "read_parquet([" + ",".join(f"'{p}'" for p in files) + "])"
    t0 = time.time()
    con.execute(build_sql(src, where, out))
    rows = con.execute(f"SELECT count(*), min(year), max(year), sum(m_n), sum(t_n), sum(a_n), "
                       f"sum(v_n), sum(n20_n) FROM '{out}'").fetchall()[0]
    print(f"rows {rows[0]:,} years {rows[1]}-{rows[2]} | MODIS {rows[3]:,} (Terra {rows[4]:,}, "
          f"Aqua {rows[5]:,}) | S-NPP {rows[6]:,} | NOAA-20 {rows[7]:,} | {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

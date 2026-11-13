"""Query engine over the harmonized dataset (DuckDB, read-only).

Area spec (dict):
  {"type": "bbox", "west": .., "south": .., "east": .., "north": ..}
  {"type": "polygon", "coordinates": [[lon, lat], ...]}           (GeoJSON ring)
  {"type": "point", "lon": .., "lat": .., "radius_km": ..}
  {"type": "country", "name": "Brazil"}
A 0.25deg cell belongs to the area when its centre is inside it.
"""
import io, json, math, os, threading
from pathlib import Path
import duckdb, numpy as np, pandas as pd

DATA = Path(os.environ.get("BRASA_DATA", Path.home() / "firms/derived"))
APP = DATA / "app"
_local = threading.local()
SRC_LABEL = {"MT": "MODIS Terra → VIIRS-eq", "MTA": "MODIS Terra+Aqua → VIIRS-eq",
             "VS": "VIIRS S-NPP observed", "VN": "VIIRS NOAA-20 observed (S-NPP gap)"}


def con():
    c = getattr(_local, "con", None)
    if c is None:
        c = _local.con = duckdb.connect(config={"threads": 4})
    return c


def _centres(cells):
    row, col = cells // 1440, cells % 1440
    return (col + 0.5) / 4 - 180, (row + 0.5) / 4 - 90  # lon, lat


def _bbox_cells(w, s, e, n):
    r0, r1 = max(int(math.floor((s + 90) * 4)), 0), min(int(math.floor((n + 90) * 4)), 719)
    if w <= e:
        cols = np.arange(max(int(math.floor((w + 180) * 4)), 0), min(int(math.floor((e + 180) * 4)), 1439) + 1)
    else:  # crosses the antimeridian
        cols = np.r_[np.arange(int(math.floor((w + 180) * 4)), 1440), np.arange(0, int(math.floor((e + 180) * 4)) + 1)]
    rows = np.arange(r0, r1 + 1)
    return (rows[:, None] * 1440 + cols[None, :]).ravel()


def _in_poly(x, y, ring):
    ring = np.asarray(ring, dtype=float)
    inside = np.zeros(len(x), bool)
    xj, yj = ring[-1]
    for xi, yi in ring:
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        inside ^= cross
        xj, yj = xi, yi
    return inside


def area_cells(area):
    t = area["type"]
    if t == "bbox":
        cells = _bbox_cells(area["west"], area["south"], area["east"], area["north"])
        lon, lat = _centres(cells)
        keep = (lat >= area["south"]) & (lat <= area["north"])
        keep &= ((lon >= area["west"]) & (lon <= area["east"])) if area["west"] <= area["east"] \
            else ((lon >= area["west"]) | (lon <= area["east"]))
        return cells[keep]
    if t == "polygon":
        ring = area["coordinates"]
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        cells = _bbox_cells(min(xs), min(ys), max(xs), max(ys))
        lon, lat = _centres(cells)
        return cells[_in_poly(lon, lat, ring)]
    if t == "point":
        rk = float(area.get("radius_km", 25))
        dlat = rk / 111.0
        dlon = rk / (111.0 * max(math.cos(math.radians(area["lat"])), 0.01))
        cells = _bbox_cells(area["lon"] - dlon, area["lat"] - dlat, area["lon"] + dlon, area["lat"] + dlat)
        lon, lat = _centres(cells)
        dist = 111.0 * np.hypot(lat - area["lat"], (lon - area["lon"]) * np.cos(np.radians(area["lat"])))
        cells = cells[dist <= rk]
        if len(cells) == 0:  # small radius: at least the containing cell
            cells = _bbox_cells(area["lon"], area["lat"], area["lon"], area["lat"])
        return cells
    if t == "country":
        return con().execute(f"SELECT cell FROM '{APP}/cell_country.parquet' WHERE country = ?",
                             [area["name"]]).df().cell.to_numpy()
    raise ValueError(f"unknown area type {t}")


def countries():
    return con().execute(f"SELECT DISTINCT country FROM '{APP}/cell_country.parquet' ORDER BY 1").df().country.tolist()


def monthly(cells):
    """Full monthly series for the area (2000-11 .. last month), harmonized, with provenance."""
    c = con()
    c.register("sel", pd.DataFrame({"cell": np.asarray(cells, dtype="int32")}))
    q = f"""
    WITH h AS (
      SELECT year, month, sum(h) h, sum(hm) hm, sum(m) m, sum(v_snpp) v_snpp, sum(v_n20) v_n20
      FROM '{APP}/harm.parquet' JOIN sel USING (cell) GROUP BY ALL),
    fl_all AS (
      SELECT month, sum(b_ta) b_ta, sum(b_t) b_t FROM '{APP}/floor.parquet' JOIN sel USING (cell) GROUP BY ALL),
    fl_used AS (  -- floor applies only where MODIS saw nothing in that cell-month
      SELECT year, month, sum(b_ta) b_ta, sum(b_t) b_t
      FROM '{APP}/harm.parquet' JOIN sel USING (cell) JOIN '{APP}/floor.parquet' USING (cell, month)
      WHERE m > 0 GROUP BY ALL)
    SELECT mo.year, mo.month, mo.src,
      coalesce(h.h, 0) AS h_obs,
      CASE WHEN mo.src = 'MT'  THEN coalesce(fa.b_t, 0)  - coalesce(fu.b_t, 0)
           WHEN mo.src = 'MTA' THEN coalesce(fa.b_ta, 0) - coalesce(fu.b_ta, 0) ELSE 0 END AS h_floor,
      coalesce(h.hm, 0) + greatest(CASE WHEN mo.src = 'MT' THEN coalesce(fa.b_t, 0) - coalesce(fu.b_t, 0)
                                        ELSE coalesce(fa.b_ta, 0) - coalesce(fu.b_ta, 0) END, 0) AS modis_eq,
      coalesce(h.m, 0) AS modis, coalesce(h.v_snpp, 0) AS viirs_snpp, coalesce(h.v_n20, 0) AS viirs_noaa20
    FROM '{APP}/months.parquet' mo
    LEFT JOIN h USING (year, month) LEFT JOIN fl_all fa USING (month) LEFT JOIN fl_used fu USING (year, month)
    ORDER BY year, month"""
    df = c.execute(q).df()
    c.unregister("sel")
    df["hotspots"] = (df.h_obs + df.h_floor.clip(lower=0)).round(1)
    return df


# Uncertainty of harmonized months from the 2019-2024 hold-out (median |log error| -> factor)
UNCERT = {"MT": 0.27, "MTA": 0.21, "VS": 0.0, "VN": 0.09}


def calendar(area, start="2000-11", end=None, baseline=(2001, 2025)):
    cells = area_cells(area)
    df = monthly(cells)
    df["ym"] = df.year * 100 + df.month
    s = int(start.replace("-", "")); e = int(end.replace("-", "")) if end else int(df.ym.max())
    base = df[(df.year >= baseline[0]) & (df.year <= baseline[1])]
    clim = base.groupby("month").hotspots.agg(
        median="median", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9), mean="mean")
    df = df.join(clim, on="month")
    df["anomaly"] = (df.hotspots - df["median"]).round(1)
    df["anomaly_pct"] = np.where(df["median"] > 0, 100 * (df.hotspots / df["median"] - 1), np.nan).round(1)
    df["z_log"] = (np.log1p(df.hotspots) - np.log1p(df["median"])).round(3)
    df["rel_uncert"] = df.src.map(UNCERT)
    sel = df[(df.ym >= s) & (df.ym <= e)].copy()
    yearly = sel.groupby("year").hotspots.sum()
    full_years = sel.groupby("year").month.count()
    yearly_full = yearly[full_years == 12]
    moy = sel.groupby("month").hotspots.median()
    last = sel.iloc[-1] if len(sel) else None
    summary = dict(
        n_cells=int(len(cells)), area_km2=round(float(_cells_area(cells)), 0),
        bbox=_bbox(cells),
        total=round(float(sel.hotspots.sum()), 0),
        peak_month=int(moy.idxmax()) if len(moy) and moy.max() > 0 else None,
        season_months=[int(m) for m in moy[moy >= 0.5 * moy.max()].index] if len(moy) and moy.max() > 0 else [],
        worst_year=int(yearly_full.idxmax()) if len(yearly_full) else None,
        worst_year_total=round(float(yearly_full.max()), 0) if len(yearly_full) else None,
        quietest_year=int(yearly_full.idxmin()) if len(yearly_full) else None,
        worst_month=dict(year=int(sel.loc[sel.hotspots.idxmax(), "year"]),
                         month=int(sel.loc[sel.hotspots.idxmax(), "month"]),
                         hotspots=float(sel.hotspots.max())) if len(sel) and sel.hotspots.max() > 0 else None,
        latest=dict(year=int(last.year), month=int(last.month), hotspots=float(last.hotspots),
                    median=float(last["median"]), anomaly_pct=None if pd.isna(last.anomaly_pct) else float(last.anomaly_pct),
                    rank=int((df[(df.month == last.month)].hotspots > last.hotspots).sum() + 1),
                    of=int((df.month == last.month).sum())) if last is not None else None,
        trend_pct_per_decade=_trend(yearly_full),
    )
    cols = ["year", "month", "src", "hotspots", "h_floor", "modis", "modis_eq", "viirs_snpp", "viirs_noaa20",
            "median", "p10", "p90", "anomaly", "anomaly_pct", "z_log", "rel_uncert"]
    return sel[cols], clim.reset_index(), yearly.reset_index(), summary


def _trend(yearly):
    if len(yearly) < 5 or yearly.sum() <= 0:
        return None
    x = yearly.index.to_numpy(float); y = np.log1p(yearly.to_numpy(float))
    b = np.polyfit(x, y, 1)[0]
    return round(100 * (math.exp(10 * b) - 1), 1)


def _bbox(cells):
    if len(cells) == 0:
        return None
    lon, lat = _centres(np.asarray(cells))
    return [float(lon.min() - 0.125), float(lat.min() - 0.125), float(lon.max() + 0.125), float(lat.max() + 0.125)]


def _cells_area(cells):
    lat = (np.asarray(cells) // 1440 + 0.5) / 4 - 90
    return float((np.cos(np.radians(lat)) * (111.32 * 0.25) ** 2).sum())


def points(area, start, end, sensors=("modis", "viirs-snpp", "viirs-jpss1"), limit=500_000):
    """Raw FIRMS detections in the area/date range (for CSV export)."""
    cells = area_cells(area)
    lon, lat = _centres(cells)
    w, e_, s, n = lon.min() - 0.125, lon.max() + 0.125, lat.min() - 0.125, lat.max() + 0.125
    y0, y1 = int(start[:4]), int(end[:4])
    files = [str(DATA / "points" / f"{sn}_{y}.parquet") for sn in sensors for y in range(y0, y1 + 1)
             if (DATA / "points" / f"{sn}_{y}.parquet").exists()]
    if not files:
        return pd.DataFrame()
    c = con()
    c.register("sel", pd.DataFrame({"cell": np.asarray(cells, dtype="int32")}))
    df = c.execute(f"""
      SELECT p.* FROM read_parquet(?) p
      JOIN sel ON sel.cell = LEAST(CAST(floor((p.lat+90)*4) AS INTEGER),719)*1440
                           + LEAST(CAST(floor((p.lon+180)*4) AS INTEGER),1439)
      WHERE p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?
        AND p.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
      ORDER BY p.date, p.time LIMIT ?""",
        [files, s, n, w, e_, f"{start}-01", _month_end(end), limit]).df()
    c.unregister("sel")
    return df


def _month_end(ym):
    y, m = map(int, ym.split("-"))
    import calendar as cal
    return f"{y}-{m:02d}-{cal.monthrange(y, m)[1]}"


def meta():
    m = json.loads((APP / "model.json").read_text())
    v = DATA / "validation_TA_train2018_test2019-2024.csv"
    if not v.exists():
        v = DATA / "validation_train2018_test2019-2024.csv"
    if v.exists():
        m["validation"] = pd.read_csv(v).to_dict("records")
    mo = con().execute(f"SELECT * FROM '{APP}/months.parquet' ORDER BY year, month").df()
    m["first_month"] = f"{int(mo.year.iloc[0])}-{int(mo.month.iloc[0]):02d}"
    m["last_month"] = f"{int(mo.year.iloc[-1])}-{int(mo.month.iloc[-1]):02d}"
    m["source_labels"] = SRC_LABEL
    return m

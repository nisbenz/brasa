"""MODIS detections + U-Net prediction -> synthetic VIIRS S-NPP detections in the FIRMS schema.

Usage: python -I ml/to_viirs.py <derived_dir> <stack_dir> <model.pt> --years 2005[-2007]
           [--months 2005-08,2005-09] [--bbox W,S,E,N] [--cal <json>] [--out <dir>]
           [--format csv|parquet] [--draw round|nb] [--min-mu 0.05] [--seed 0]
Needs  <stack_dir>       from ml/build_tensor.py covering the requested months (any years: the
                         inputs are MODIS only)
       <derived>/points/modis_<year>.parquet, <derived>/app/cell_country.parquet
       ml/viirs_calibration.json from ml/viirs_calibrate.py
Writes <out>/viirs-snpp-unet/<year>/viirs-snpp-unet_<year>_<Country>.csv   (FIRMS archive layout)
       or <out>/viirs-snpp-unet/<year>/viirs-snpp-unet_<year>_<MM>.parquet  (--format parquet)
       <out>/viirs-snpp-unet/<year>/cells_<year>.parquet   per cell-month: MODIS count, mu, alpha, n
       <out>/viirs-snpp-unet/manifest.json

Columns, in FIRMS VIIRS order: latitude, longitude, bright_ti4, scan, track, acq_date, acq_time,
satellite, instrument, confidence, version, bright_ti5, frp, daynight, type.

How a cell-month becomes points:
1. Count. The U-Net gives mu (expected good S-NPP detections) and alpha (NB overdispersion).
   --draw round: floor(mu) plus one more with probability frac(mu), so totals match mu in
   expectation and stay close to it at every scale. --draw nb: one NB(mu, alpha) draw, i.e. one
   plausible month with the model's own scatter. Cells with mu < --min-mu, or outside any
   country, get none (the grid includes ocean, where the baseline is small but not zero).
2. Templates. Each synthetic detection copies a good MODIS detection of the same cell-month,
   picked at random (with replacement, VIIRS typically sees ~4x more). It keeps the template's
   date and day/night flag and lands at a random spot inside the template's MODIS footprint
   (scan x track km), clipped to the cell.
3. Attributes. frp, bright_ti4, scan/track and confidence come from monotone quantile mapping:
   a MODIS value at rank u in its distribution becomes the S-NPP value at rank u (+ small
   jitter so copies of one template differ). bright_ti5 is drawn from S-NPP's ti5 | ti4 bin.
   acq_time is drawn from S-NPP's local overpass hours for day or night and converted to UTC
   (which can move acq_date by a day, as in real FIRMS files).
4. Floor fires. Cells where MODIS saw nothing but the model still expects VIIRS fire (the small
   fires MODIS misses) get detections placed uniformly in the cell and month, with attributes
   drawn from S-NPP's own distribution in such cell-months.
version is "unet-v1" on every row, so synthetic detections never pass for real ones.
"""
import argparse
import calendar
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Stack, UNet, predict_month

SENSOR = "viirs-snpp-unet"
VERSION = "unet-v1"
COLUMNS = ["latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
           "satellite", "instrument", "confidence", "version", "bright_ti5", "frp", "daynight", "type"]
MGOOD = "type = 0 AND TRY_CAST(confidence AS INTEGER) >= 30"
CELL = ("(LEAST(CAST(floor((lat+90)*4) AS INTEGER),719)*1440 + "
        "LEAST(CAST(floor((lon+180)*4) AS INTEGER),1439))")
KM_PER_DEG = 111.32
JITTER = 0.03            # quantile-space jitter for copies of the same template


class Mapper:
    """Quantile maps from viirs_calibration.json."""

    def __init__(self, cal, rng):
        self.c, self.rng = cal, rng
        self.q = np.array(cal["q"])

    def rank(self, dn, field, v):
        """u in [0, 1] of MODIS value v in its own day/night distribution."""
        u = np.empty(len(v))
        for d in ("D", "N"):
            k = dn == d
            u[k] = np.interp(v[k], self.c["modis"][d][field], self.q)
        return u

    def value(self, table, dn, field, u):
        out = np.empty(len(u))
        for d in ("D", "N"):
            k = dn == d
            out[k] = np.interp(u[k], self.q, self.c[table][d][field])
        return out

    def jitter(self, u, sd=JITTER):
        return np.clip(u + self.rng.normal(0, sd, len(u)), 0, 1)

    def ti5(self, dn, ti4):
        b = np.digitize(ti4, self.c["ti5_edges"])
        u = self.rng.random(len(ti4))
        out = np.empty(len(ti4))
        for d in ("D", "N"):
            tab = np.array(self.c["ti5"][d])
            for k in np.unique(b[dn == d]):
                m = (dn == d) & (b == k)
                out[m] = np.interp(u[m], self.q, tab[k])
        return out

    def local_hour(self, dn):
        u = self.rng.random(len(dn))
        out = np.empty(len(dn))
        for d in ("D", "N"):
            k = dn == d
            out[k] = np.interp(u[k], self.q, self.c["local_hour"][d])
        return out


def draw_counts(mu, alpha, mode, rng):
    if mode == "nb":
        r = 1 / np.maximum(alpha, 1e-6)
        return rng.negative_binomial(r, r / (r + mu))
    base = np.floor(mu)
    return (base + (rng.random(mu.shape) < mu - base)).astype(np.int64)


def modis_templates(con, derived, y, m, where):
    return con.execute(f"""
        SELECT {CELL} AS cell, lat, lon, date, daynight, frp, bright, scan, track,
               TRY_CAST(confidence AS INTEGER)::DOUBLE AS conf
        FROM '{derived}/points/modis_{y}.parquet'
        WHERE {MGOOD} AND month(date) = {m} AND daynight IN ('D', 'N') AND {where}
        ORDER BY cell""").df()


def synthesize(cells, n, tpl, y, m, mp, rng, cal):
    """cells/n: arrays of cell ids and detection counts. tpl: MODIS templates sorted by cell."""
    tcell = tpl.cell.to_numpy()
    lo, hi = np.searchsorted(tcell, cells, "left"), np.searchsorted(tcell, cells, "right")
    k = hi - lo
    has = k > 0
    parts = []

    # 2-3: detections copied from MODIS templates
    rep = np.repeat(np.arange(len(cells))[has], n[has])
    if len(rep):
        idx = lo[rep] + (rng.random(len(rep)) * k[rep]).astype(np.int64)
        t = tpl.iloc[idx]
        dn = t.daynight.to_numpy()
        lat0, lon0 = t.lat.to_numpy(np.float64), t.lon.to_numpy(np.float64)
        u_scan = mp.jitter(mp.rank(dn, "scan", t.scan.to_numpy(np.float64)))
        u_conf = mp.jitter(mp.rank(dn, "conf", t.conf.to_numpy(np.float64)))
        h_share = np.where(dn == "D", cal["h_share"]["D"], cal["h_share"]["N"])
        dlat = (rng.random(len(t)) - 0.5) * t.track.to_numpy(np.float64) / KM_PER_DEG
        dlon = ((rng.random(len(t)) - 0.5) * t.scan.to_numpy(np.float64)
                / (KM_PER_DEG * np.maximum(np.cos(np.radians(lat0)), 0.05)))
        parts.append(dict(
            cell=cells[rep], lat=lat0 + dlat, lon=lon0 + dlon, dn=dn,
            date=t.date.to_numpy("datetime64[D]"),
            frp=mp.value("viirs", dn, "frp", mp.jitter(mp.rank(dn, "frp", t.frp.to_numpy(np.float64)))),
            ti4=mp.value("viirs", dn, "bright_ti4",
                         mp.jitter(mp.rank(dn, "bright", t.bright.to_numpy(np.float64)))),
            scan=mp.value("viirs", dn, "scan", u_scan), track=mp.value("viirs", dn, "track", u_scan),
            h=u_conf > 1 - h_share))

    # 4: floor fires (no MODIS detection in the cell-month)
    rep = np.repeat(np.arange(len(cells))[~has], n[~has])
    if len(rep):
        c = cells[rep]
        N = len(c)
        dn = np.where(rng.random(N) < cal["floor_day_share"], "D", "N")
        u = rng.random(N)
        u_scan = rng.random(N)
        days = calendar.monthrange(y, m)[1]
        h_share = np.where(dn == "D", cal["floor_h_share"]["D"], cal["floor_h_share"]["N"])
        parts.append(dict(
            cell=c, lat=(c // 1440 + rng.random(N)) / 4 - 90, lon=(c % 1440 + rng.random(N)) / 4 - 180,
            dn=dn, date=np.datetime64(f"{y}-{m:02d}-01") + rng.integers(0, days, N).astype("timedelta64[D]"),
            frp=mp.value("floor", dn, "frp", u),
            ti4=mp.value("floor", dn, "bright_ti4", mp.jitter(u, 0.15)),
            scan=mp.value("floor", dn, "scan", u_scan), track=mp.value("floor", dn, "track", u_scan),
            h=rng.random(N) < h_share))
    if not parts:
        return None
    p = {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}

    # keep every point inside its own cell, so re-aggregating reproduces the counts
    eps = 1e-5
    rlat, rlon = (p["cell"] // 1440) / 4 - 90, (p["cell"] % 1440) / 4 - 180
    lat = np.clip(p["lat"], rlat + eps, rlat + 0.25 - eps)
    lon = np.clip(p["lon"], rlon + eps, rlon + 0.25 - eps)
    utc = mp.local_hour(p["dn"]) - lon / 15
    shift = np.floor(utc / 24).astype(np.int64)
    utc = utc - 24 * shift
    hh = np.minimum((utc * 60).astype(np.int64), 24 * 60 - 1)
    df = pd.DataFrame({
        "cell": p["cell"],
        "latitude": lat.round(5), "longitude": lon.round(5),
        "bright_ti4": p["ti4"].round(2), "scan": p["scan"].round(2), "track": p["track"].round(2),
        "acq_date": (p["date"] + shift.astype("timedelta64[D]")).astype(str),
        "acq_time": [f"{v // 60:02d}{v % 60:02d}" for v in hh],
        "satellite": "N", "instrument": "VIIRS",
        "confidence": np.where(p["h"], "h", "n"), "version": VERSION,
        "bright_ti5": mp.ti5(p["dn"], p["ti4"]).round(2), "frp": np.maximum(p["frp"], 0).round(2),
        "daynight": p["dn"], "type": 0,
    })
    return df.sort_values(["acq_date", "acq_time"], kind="stable")


def month_list(stack, a):
    if a.months:
        want = {tuple(map(int, s.split("-"))) for s in a.months.split(",")}
    else:
        y0, _, y1 = a.years.partition("-")
        want = {(y, m) for y in range(int(y0), int(y1 or y0) + 1) for m in range(1, 13)}
    idx = [t for t, ym in enumerate(stack.months) if ym in want]
    missing = sorted(want - {stack.months[t] for t in idx})
    if missing and a.months:
        raise SystemExit(f"months not in stack {stack.path}: {missing}")
    if not idx:
        raise SystemExit(f"no requested month is in stack {stack.path} "
                         f"({stack.meta['months'][0]}..{stack.meta['months'][-1]})")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path); ap.add_argument("stack"); ap.add_argument("model")
    ap.add_argument("--years", default=None); ap.add_argument("--months", default=None)
    ap.add_argument("--bbox", default=None, help="W,S,E,N degrees")
    ap.add_argument("--cal", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--draw", choices=["round", "nb"], default="round")
    ap.add_argument("--min-mu", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not (a.years or a.months):
        ap.error("give --years or --months")
    rng = np.random.default_rng(a.seed)
    out = (a.out or a.derived / "ml" / "synthetic") / SENSOR
    cal_path = a.cal or a.derived / "ml" / "viirs_calibration.json"
    mp = Mapper(json.loads(cal_path.read_text()), rng)
    cal = mp.c
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    s = Stack(a.stack)
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    if ck["n_in"] != s.n_in:
        raise SystemExit(f"model expects {ck['n_in']} inputs, stack gives {s.n_in}")
    model = UNet(ck["n_in"], ck["width"]).to(dev); model.load_state_dict(ck["state"]); model.eval()
    months = month_list(s, a)

    cc = pd.read_parquet(next(p for p in (a.derived / "app" / "cell_country.parquet",
                                          a.derived / "cell_country.parquet") if p.exists()))
    country = pd.Series(cc.country.to_numpy(), index=cc.cell.to_numpy())
    r0, r1, c0, c1 = (s.meta[k] for k in ("r0", "r1", "c0", "c1"))
    rows, cols = np.mgrid[r0:r1, c0:c1]
    grid_cell = rows * 1440 + cols
    keep = np.isin(grid_cell, country.index.to_numpy())
    where = "TRUE"
    if a.bbox:
        w, so, e, no = map(float, a.bbox.split(","))
        lat_c, lon_c = (rows + 0.5) / 4 - 90, (cols + 0.5) / 4 - 180
        keep &= (lat_c >= so) & (lat_c < no) & (lon_c >= w) & (lon_c < e)
        where = f"lat >= {so} AND lat < {no} AND lon >= {w} AND lon < {e}"

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    years_done, cell_tabs, written = set(), {}, {}
    for t in months:
        y, m = s.months[t]
        ydir = out / str(y)
        if y not in years_done:                      # fresh output for every year we touch
            ydir.mkdir(parents=True, exist_ok=True)
            for f in ydir.glob(f"{SENSOR}_{y}_*"):
                f.unlink()
            years_done.add(y)
        mu, alpha = predict_month(model, s, t, dev, with_alpha=True)
        mu = np.where(keep & (mu >= a.min_mu), mu, 0)
        n = draw_counts(mu, alpha, a.draw, rng)
        sel = n > 0
        cells = grid_cell[sel]
        tpl = modis_templates(con, a.derived, y, m, where)
        df = synthesize(cells, n[sel], tpl, y, m, mp, rng, cal)
        m_n = tpl.cell.value_counts()
        live = (mu > 0) | np.isin(grid_cell, m_n.index.to_numpy())
        cell_tabs.setdefault(y, []).append(pd.DataFrame({
            "cell": grid_cell[live], "year": y, "month": m,
            "m_n": m_n.reindex(grid_cell[live]).fillna(0).to_numpy(np.int32),
            "mu": mu[live].astype(np.float32), "alpha": alpha[live].astype(np.float32),
            "n": n[live].astype(np.int32)}))
        if df is None:
            print(f"{y}-{m:02d}: no detections", flush=True)
            continue
        if a.format == "parquet":
            df[COLUMNS].to_parquet(ydir / f"{SENSOR}_{y}_{m:02d}.parquet", index=False)
        else:
            names = country.reindex(df.cell.to_numpy()).to_numpy()
            for name, g in df.groupby(names, sort=False):
                f = ydir / f"{SENSOR}_{y}_{str(name).replace(' ', '_')}.csv"
                g[COLUMNS].to_csv(f, mode="a", header=not f.exists(), index=False)
        written[y] = written.get(y, 0) + len(df)
        print(f"{y}-{m:02d}: MODIS {len(tpl):,} -> expected {mu.sum():,.0f} -> wrote {len(df):,} "
              f"({(~np.isin(cells, tpl.cell.to_numpy())).sum():,} floor cells)", flush=True)
    for y, tabs in cell_tabs.items():
        pd.concat(tabs, ignore_index=True).to_parquet(out / str(y) / f"cells_{y}.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(
        model=str(a.model), epoch=ck.get("epoch"), stack=str(a.stack), calibration=str(cal_path),
        calibration_year=cal["year"], draw=a.draw, min_mu=a.min_mu, seed=a.seed, bbox=a.bbox,
        format=a.format, version=VERSION, columns=COLUMNS, rows_per_year=written), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

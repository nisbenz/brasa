"""Build the harmonized VIIRS(S-NPP)-equivalent cell-month dataset.

Usage: python -I 04_build.py <derived_dir>
Writes <derived>/app/
  harm.parquet   cell, year, month, h, src, m, v_snpp, v_n20
                 h = VIIRS-equivalent count for the cell-month (floor NOT included)
  floor.parquet  cell, month, b_ta, b_t   small-fire floor of each active cell,
                 added at query time for MODIS-derived months where the cell had m == 0
  months.parquet year, month, src, modis_days_t, modis_days_a, cov_factor
  model.json     fitted ratios summary + source rules
Sources: MT  = harmonized MODIS Terra-only     (2000-11 .. 2002-06)
         MTA = harmonized MODIS Terra+Aqua     (2002-07 .. 2012-01)
         VS  = observed VIIRS S-NPP            (2012-02 ..)
         VN  = observed VIIRS NOAA-20          (S-NPP outage months)
"""
import calendar, json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from harmonize import fit, ratio, parents

d = Path(sys.argv[1]); out = d / "app"; out.mkdir(exist_ok=True)
con = duckdb.connect()
cm = con.sql(f"""
  SELECT cell, year, month,
    coalesce(sum(n) FILTER (WHERE sensor='modis'), 0)::DOUBLE AS m_ta,
    coalesce(sum(n) FILTER (WHERE sensor='modis' AND sat='Terra'), 0)::DOUBLE AS m_t,
    coalesce(sum(n) FILTER (WHERE sensor='viirs-snpp'), 0)::DOUBLE AS v_snpp,
    coalesce(sum(n) FILTER (WHERE sensor='viirs-jpss1'), 0)::DOUBLE AS v_n20
  FROM '{d}/cell_month.parquet' GROUP BY ALL""").df()

# ---- per-month source + MODIS day coverage --------------------------------------------
days = con.sql(f"""
  SELECT year(date) AS "year", month(date) AS "month",
    count(DISTINCT date) FILTER (WHERE satellite='Terra') dt,
    count(DISTINCT date) FILTER (WHERE satellite='Aqua') da
  FROM '{d}/points/modis_*.parquet' GROUP BY ALL""").df()
tot = cm.groupby(["year", "month"])[["v_snpp", "v_n20"]].sum().reset_index().astype({"year": "int64", "month": "int64"})
days = days.astype({"year": "int64", "month": "int64"})
mo = tot.merge(days, on=["year", "month"], how="outer").fillna(0).sort_values(["year", "month"])
mo = mo[(mo.year.astype("int64") * 100 + mo.month) >= 200011]


def src(r):
    ym = int(r.year) * 100 + int(r.month)
    if ym <= 200206:
        return "MT"
    if ym <= 201201:
        return "MTA"
    if r.v_n20 > 0 and r.v_snpp < 0.85 * r.v_n20:
        return "VN"
    return "VS"


mo["src"] = mo.apply(src, axis=1)
mo["ndays"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(mo.year, mo.month)]
used_days = np.where(mo.src == "MT", mo.dt, np.minimum(mo.dt, mo.da).where(mo.da > 0, mo.dt))
mo["cov_factor"] = np.where(mo.src.isin(["MT", "MTA"]),
                            mo.ndays / np.maximum(used_days, 1), 1.0).round(3)
mo = mo.rename(columns={"dt": "modis_days_t", "da": "modis_days_a"})

# ---- fit models on the overlap, skipping bad months ------------------------------------
bad = {(2012, 1)} | set(map(tuple, mo.loc[mo.src == "VN", ["year", "month"]].to_numpy()))
ta_q = mo[(mo.modis_days_a > 0)]
r_at = (cm.groupby(["year", "month"]).m_ta.sum() - cm.groupby(["year", "month"]).m_t.sum()) \
    / cm.groupby(["year", "month"]).m_t.sum()
med = r_at[r_at.index.get_level_values(0) >= 2012].median()
bad |= {k for k, v in r_at.items() if k[0] >= 2012 and (v < 0.6 * med or v > 1.6 * med)}  # Terra/Aqua outages


def train_set(mcol, y0, y1):
    t = cm[(cm.year >= y0) & (cm.year <= y1)]
    t = t[[(y, m) not in bad for y, m in zip(t.year, t.month)]]
    t = t.rename(columns={mcol: "M", "v_snpp": "V"})[["cell", "year", "month", "M", "V"]]
    return t[(t.M > 0) | (t.V > 0)]


model_ta = fit(train_set("m_ta", 2012, 2024), a=20, seasonal=True, floor=True)
model_t = fit(train_set("m_t", 2012, 2019), a=20, seasonal=True, floor=True)

# ---- harmonized values -----------------------------------------------------------------
cm = cm.astype({"year": "int64", "month": "int64"}).merge(mo[["year", "month", "src", "cov_factor"]], on=["year", "month"], how="inner")
h = np.zeros(len(cm))
for s, model, col in (("MT", model_t, "m_t"), ("MTA", model_ta, "m_ta")):
    sel = (cm.src == s).to_numpy() & (cm[col] > 0).to_numpy()
    M = (cm.loc[sel, col] * cm.loc[sel, "cov_factor"]).to_numpy()
    h[sel] = ratio(model, cm.cell.to_numpy()[sel], cm.month.to_numpy()[sel], M) * M
h[(cm.src == "VS").to_numpy()] = cm.v_snpp[cm.src == "VS"]
h[(cm.src == "VN").to_numpy()] = cm.v_n20[cm.src == "VN"]
cm["h"] = h.astype("float32")
cm["m"] = np.where(cm.src == "MT", cm.m_t, cm.m_ta)
# hm: MODIS converted to VIIRS terms for EVERY month (incl. the VIIRS era), so the app can show
# converted MODIS next to observed VIIRS in the overlap (floor added at query time, as for h)
hm = h.copy()
sel = (~cm.src.isin(["MT", "MTA"])).to_numpy() & (cm.m_ta > 0).to_numpy()
hm[(~cm.src.isin(["MT", "MTA"])).to_numpy()] = 0
hm[sel] = ratio(model_ta, cm.cell.to_numpy()[sel], cm.month.to_numpy()[sel], cm.m_ta.to_numpy()[sel]) * cm.m_ta.to_numpy()[sel]
cm["hm"] = hm.astype("float32")
harm = cm[["cell", "year", "month", "h", "hm", "src", "m", "v_snpp", "v_n20"]]
harm = harm[(harm.h > 0) | (harm.m > 0) | (harm.v_snpp > 0) | (harm.v_n20 > 0)]
harm = harm.astype({"cell": "int32", "year": "int16", "month": "int8",
                    "m": "int32", "v_snpp": "int32", "v_n20": "int32"})
con.register("harm", harm)
con.execute(f"COPY (SELECT * FROM harm ORDER BY cell, year, month) TO '{out}/harm.parquet' (FORMAT parquet, COMPRESSION zstd)")

# ---- floor per active cell -------------------------------------------------------------
def cell_floor(model, name):
    act = model["active"]
    f = pd.DataFrame({"cell": act.index.to_numpy(), "p1": act.to_numpy()}).merge(
        model["b"].rename(name).reset_index(), on="p1")
    return f[["cell", "month", name]]


fl = cell_floor(model_ta, "b_ta").merge(cell_floor(model_t, "b_t"), on=["cell", "month"], how="outer").fillna(0)
fl = fl[(fl.b_ta > 0) | (fl.b_t > 0)].astype({"cell": "int32", "month": "int8", "b_ta": "float32", "b_t": "float32"})
con.register("fl", fl)
con.execute(f"COPY (SELECT * FROM fl ORDER BY cell, month) TO '{out}/floor.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.register("mo", mo[["year", "month", "src", "modis_days_t", "modis_days_a", "cov_factor"]])
con.execute(f"COPY mo TO '{out}/months.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT * FROM '{d}/cell_country.parquet') TO '{out}/cell_country.parquet' (FORMAT parquet)")

meta = dict(
    reference="VIIRS S-NPP 375 m, type=0 (vegetation fire), confidence nominal/high",
    modis_filter="MODIS C6.1, type=0, confidence >= 30",
    grid_deg=0.25, global_ratio_ta=round(model_ta["kg"], 3), global_ratio_t=round(model_t["kg"], 3),
    excluded_training_months=sorted(f"{y}-{m:02d}" for y, m in bad),
    source_counts=mo.src.value_counts().to_dict(),
    coverage_corrected_months=mo.loc[mo.cov_factor > 1.05, ["year", "month", "cov_factor"]].to_dict("records"),
)
(out / "model.json").write_text(json.dumps(meta, indent=1, default=str))
print(json.dumps(meta, indent=1, default=str))
print("harm rows", len(harm), "floor rows", len(fl))
g = harm.groupby("year").h.sum()
print((g / 1e6).round(2).to_string())

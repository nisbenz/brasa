"""Consistency check of the Terra-only model in the pre-VIIRS era.

In 2003-2010 both Terra and Aqua flew, so we can harmonize the same months twice:
with the Terra-only model (as used for 2000-11..2002-06) and with the Terra+Aqua model.
Agreement means the Terra-only years are on the same scale as the rest of the record.
Usage: python -I 05_check_terra.py <derived_dir>
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from harmonize import fit, predict_regions

d = Path(sys.argv[1])
con = duckdb.connect()
cm = con.sql(f"""SELECT cell, year::INT AS year, month::INT AS month,
    coalesce(sum(n) FILTER (WHERE sensor='modis'),0)::DOUBLE m_ta,
    coalesce(sum(n) FILTER (WHERE sensor='modis' AND sat='Terra'),0)::DOUBLE m_t,
    coalesce(sum(n) FILTER (WHERE sensor='viirs-snpp'),0)::DOUBLE V
    FROM '{d}/cell_month.parquet' GROUP BY ALL""").df()
cc = con.sql(f"SELECT cell, country FROM '{d}/cell_country.parquet'").df().set_index("cell").country
bad = {(2012, 1), (2022, 4), (2022, 8), (2024, 7)}


def tr(col, y0, y1):
    t = cm[(cm.year.between(y0, y1)) & np.array([(y, m) not in bad for y, m in zip(cm.year, cm.month)])]
    t = t.rename(columns={col: "M"})[["cell", "year", "month", "M", "V"]]
    return t[(t.M > 0) | (t.V > 0)]


mta = fit(tr("m_ta", 2012, 2024)); mt = fit(tr("m_t", 2012, 2019))
era = cm[cm.year.between(2003, 2010)]
fn = lambda c: cc.reindex(c).fillna("?").to_numpy()
a = predict_regions(mta, era.rename(columns={"m_ta": "M"})[["cell", "year", "month", "M"]], fn)
b = predict_regions(mt, era.rename(columns={"m_t": "M"})[["cell", "year", "month", "M"]], fn)
y = pd.concat([a.rename("TA"), b.rename("T")], axis=1).fillna(0).groupby(["region", "year"]).sum()
tot = y.groupby("year").sum()
print("global yearly  T-only / T+A:", (tot["T"] / tot["TA"]).round(3).to_dict())
big = y.groupby("region").TA.sum().nlargest(15).index
yy = y.loc[big]
le = np.log((yy["T"] + 1) / (yy["TA"] + 1))
print("15 biggest countries, country-year median |log diff|:", round(float(np.median(np.abs(le))), 3))
print((y.loc[big].groupby("region").sum().eval("T/TA")).round(3).to_string())

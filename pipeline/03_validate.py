"""Temporal hold-out validation of harmonization models.

Usage: python -I 03_validate.py <derived_dir> [train_end] [test_start] [test_end]
Train on overlap years <= train_end, test on test_start..test_end against observed VIIRS S-NPP.
Scored at three region scales: 1deg boxes, 5deg boxes, countries.
"""
import sys, json
from pathlib import Path
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from harmonize import fit, predict_regions, parents

d = Path(sys.argv[1])
TR_END = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
TE0 = int(sys.argv[3]) if len(sys.argv) > 3 else 2019
TE1 = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
MSRC = sys.argv[5] if len(sys.argv) > 5 else 'TA'  # TA = Terra+Aqua, T = Terra only

con = duckdb.connect()
cm = con.sql(f"""
  SELECT cell, year, month,
         sum(n) FILTER (WHERE sensor='modis' AND ('{MSRC}'='TA' OR sat='Terra'))::DOUBLE AS M,
         sum(n) FILTER (WHERE sensor='viirs-snpp')::DOUBLE AS V
  FROM '{d}/cell_month.parquet' WHERE year BETWEEN 2012 AND {TE1} GROUP BY ALL
""").df().fillna(0)
cm = cm[(cm.M > 0) | (cm.V > 0)]
cc = con.sql(f"SELECT cell, country FROM '{d}/cell_country.parquet'").df().set_index("cell").country
train = cm[cm.year <= TR_END]
test = cm[(cm.year >= TE0) & (cm.year <= TE1)]

scales = {
    "1deg": lambda c: parents(c)[0],
    "5deg": lambda c: parents(c)[1],
    "country": lambda c: cc.reindex(c).fillna("?").to_numpy(),
}


def score(pred, obs):
    j = pd.concat([pred.rename("p"), obs.rename("o")], axis=1).fillna(0)
    j = j[(j.p + j.o) > 0]
    big = j[(j.p + j.o) >= 20]
    le = np.log((big.p + 1) / (big.o + 1))
    # calendar anomaly agreement: per region & calendar month, anomaly vs the test-period median
    k = j.reset_index()
    tot = k.groupby("region").o.transform("sum")
    k = k[tot >= 200]
    full = k.set_index(["region", "year", "month"])
    regions = full.index.get_level_values(0).unique()
    grid = pd.MultiIndex.from_product([regions, range(TE0, TE1 + 1), range(1, 13)],
                                      names=["region", "year", "month"])
    full = full.reindex(grid).fillna(0).reset_index()
    for col in ("p", "o"):
        med = full.groupby(["region", "month"])[col].transform("median")
        full[col + "_an"] = np.log((full[col] + 1) / (med + 1))
    return dict(
        bias_pct=round(100 * (j.p.sum() / j.o.sum() - 1), 2),
        med_abs_log_err=round(float(np.median(np.abs(le))), 3),
        within_25pct=round(float((np.abs(le) < np.log(1.25)).mean()), 3),
        r_log=round(float(np.corrcoef(np.log1p(j.p), np.log1p(j.o))[0, 1]), 4),
        anomaly_r=round(float(np.corrcoef(full.p_an, full.o_an)[0, 1]), 4),
        anomaly_sign_agree=round(float((np.sign(full.p_an) == np.sign(full.o_an))[
            np.abs(full.o_an) > np.log(1.5)].mean()), 3),
    )


models = {
    "M0 global ratio": dict(a=1e12, seasonal=False, floor=False),
    "M1 spatial (a=20)": dict(a=20, seasonal=False, floor=False),
    "M2 spatial+season (a=20)": dict(a=20, seasonal=True, floor=False),
    "M3 +floor (a=5)": dict(a=5, seasonal=True, floor=True),
    "M3 +floor (a=20)": dict(a=20, seasonal=True, floor=True),
    "M3 +floor (a=100)": dict(a=100, seasonal=True, floor=True),
    "M4 +floor+intensity (a=20)": dict(a=20, seasonal=True, floor=True, intensity=True),
}
modis_test = test[test.M > 0][["cell", "year", "month", "M"]]
results = {}
for name, kw in models.items():
    model = fit(train, **kw)
    for sc, fn in scales.items():
        pred = predict_regions(model, modis_test, fn)
        if "b" not in model:
            pass
        o = test.assign(region=fn(test.cell.to_numpy())).groupby(["region", "year", "month"]).V.sum()
        results[(name, sc)] = score(pred, o)
        print(name, sc, results[(name, sc)], flush=True)
# noise ceiling: a second VIIRS instrument (NOAA-20) scored against S-NPP the same way
n20 = con.sql(f"""SELECT cell, year, month, sum(n)::DOUBLE AS N FROM '{d}/cell_month.parquet'
  WHERE sensor='viirs-jpss1' AND year BETWEEN {max(TE0, 2019)} AND {TE1} GROUP BY ALL""").df()
for sc, fn in scales.items():
    t2 = test[test.year >= max(TE0, 2019)]
    o = t2.assign(region=fn(t2.cell.to_numpy())).groupby(["region", "year", "month"]).V.sum()
    p = n20.assign(region=fn(n20.cell.to_numpy())).groupby(["region", "year", "month"]).N.sum()
    results[("CEILING NOAA-20 vs S-NPP", sc)] = score(p, o)
out = pd.DataFrame(results).T
out.index.names = ["model", "scale"]
out.to_csv(d / f"validation_{MSRC}_train{TR_END}_test{TE0}-{TE1}.csv")
print(out.to_string())

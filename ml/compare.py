"""Compare every cached per-cell prediction against Brasa M3 and the NOAA-20 ceiling.

Usage: python -I ml/compare.py <derived_dir> <cells_dir>
Models: Brasa M3 refit on 2012-2018 (as in ml/evaluate.py), plus each <cells_dir>/<name>_cells.parquet
(from ml/predict_cells.py or ml/context_baseline.py). Reports, on 2019-2024 and on 2025 separately:
- pipeline/03_validate.py metrics at 1deg, 5deg and country scale
- 5deg error and bias per year (sensor drift)
- paired bootstrap over months of the 5deg median |log err| difference, with a 95% interval
- bias and cell-level error by fire regime, and in cells where MODIS saw nothing
- median monthly error in the 15 biggest fire countries
Writes scores.csv, regimes.csv, countries.csv and bootstrap.json to <cells_dir>.
"""
import itertools, json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pipeline"))
from evaluate import score
from harmonize import fit, ratio, parents

DER, D = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.load(open(DER / "ml/overlap/meta.json"))
bad = set(meta["excluded_months"])
f = duckdb.sql(f"""SELECT cell, year, month, coalesce(m_n,0)::DOUBLE M, coalesce(v_n,0)::DOUBLE V,
    coalesce(n20_n,0)::DOUBLE N FROM '{DER}/ml/features.parquet'
    WHERE year >= 2012 AND cell // 1440 >= {meta['r0']} AND cell // 1440 < {meta['r1']}""").df()
f = f[[f"{y}-{m:02d}" not in bad for y, m in zip(f.year, f.month)]]
KEYS = {"cell": "int32", "year": "int16", "month": "int8"}   # same key dtypes everywhere for joins
f = f.astype(KEYS)
train = f[(f.year <= 2018) & ((f.M > 0) | (f.V > 0))]
test = f[f.year >= 2019].reset_index(drop=True)
del f
brasa = fit(train, a=20, seasonal=True, floor=True)
clim = (train.groupby("cell").M.sum() / 7).astype("float32")   # fire regime: MODIS dets/yr 2012-2018
del train


def brasa_cells(model, modis):
    """Brasa M3 per cell: ratio * M where MODIS saw fire, else the 1deg floor for active cells.
    (predict_regions with one region per cell would drop cells that weren't active in training.)"""
    m = modis[modis.M > 0].copy()
    m["pred"] = ratio(model, m.cell.to_numpy(), m.month.to_numpy(), m.M.to_numpy()) * m.M
    act = model["active"]
    fl = pd.DataFrame({"cell": act.index.to_numpy(), "p1": act.to_numpy()}).merge(
        model["b"].rename("pred").reset_index(), on="p1")
    fl = fl[fl.pred > 0].drop(columns="p1").merge(pd.DataFrame({"year": modis.year.unique()}), how="cross")
    fl = fl.astype(KEYS).merge(m[["cell", "year", "month"]], how="left", indicator=True)
    fl = fl[fl._merge == "left_only"].drop(columns="_merge")   # floor only where MODIS saw nothing
    return pd.concat([m[["cell", "year", "month", "pred"]], fl], ignore_index=True).astype(KEYS | {"pred": "float32"})


SLIM = KEYS | {"pred": "float32"}
models = {"Brasa M3": brasa_cells(brasa, test[["cell", "year", "month", "M"]])}
for p in sorted(D.glob("*_cells.parquet")):   # ~18 M rows each
    models[p.name[:-len("_cells.parquet")]] = pd.read_parquet(p).astype(SLIM)
models["ceiling NOAA-20"] = test.loc[test.N > 0, ["cell", "year", "month", "N"]].rename(columns={"N": "pred"}).astype(SLIM)
learned = [k for k in models if k not in ("Brasa M3", "ceiling NOAA-20")]
print("models:", ", ".join(f"{k} ({len(v) / 1e6:.1f} M rows)" for k, v in models.items()), flush=True)

# region lookups as dense int arrays over all cells (string country keys on ~30 M rows ran Colab out of RAM)
cc = pd.read_parquet(DER / "app/cell_country.parquet")
names = np.array(sorted(cc.country.dropna().unique().tolist()) + ["?"], dtype=object)
code = np.full(720 * 1440, len(names) - 1, np.int16)
code[cc.cell.to_numpy()] = pd.Categorical(cc.country.fillna("?"), categories=names).codes
scales = {"1deg": lambda c: parents(c)[0], "5deg": lambda c: parents(c)[1], "country": lambda c: code[c]}
used = sorted(set(zip(test.year, test.month)))
periods = {"2019-2024": [x for x in used if x[0] <= 2024], "2025": [x for x in used if x[0] == 2025]}
years = {str(y): [x for x in used if x[0] == y] for y in range(2019, 2026)}


def rs(df, fn, col):
    """Sum of col per (region, year, month) without copying df."""
    v = pd.Series(df[col].to_numpy(np.float64))
    r = v.groupby([fn(df.cell.to_numpy()), df.year.to_numpy(), df.month.to_numpy()]).sum()
    return r.rename_axis(["region", "year", "month"])


R, boots = [], {}
for sc, fn in scales.items():
    obs = rs(test, fn, "V")
    agg = {k: rs(v, fn, "pred") for k, v in models.items()}
    for k, p in agg.items():
        for label, ms in (periods | years if sc == "5deg" else periods).items():
            if ms:   # anomaly metrics are undefined within a single year (NaN there); only error/bias used
                with np.errstate(invalid="ignore", divide="ignore"):
                    R.append(dict(model=k, scale=sc, period=label, **score(p, obs, ms)))
    if sc == "country":
        country_agg, country_obs = agg, obs
    if sc != "5deg":
        continue
    # paired bootstrap over months: region-months with >= 20 observed detections, 2019-2024
    j = pd.concat(agg | {"obs": obs}, axis=1).fillna(0)
    j = j[(j.obs >= 20) & (j.index.get_level_values(1) <= 2024)]
    err = {k: np.abs(np.log((j[k] + 1) / (j.obs + 1))).to_numpy() for k in agg}
    mk = (j.index.get_level_values(1).astype(int) * 12 + j.index.get_level_values(2).astype(int)).to_numpy()
    um = np.unique(mk); grp = [np.nonzero(mk == u)[0] for u in um]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([grp[i] for i in rng.integers(0, len(um), len(um))]) for _ in range(500)]
    pairs = [(k, "Brasa M3") for k in learned] + list(itertools.combinations(learned, 2))
    for x, y in pairs:
        d = [np.median(err[x][ix]) - np.median(err[y][ix]) for ix in draws]
        boots[f"{x} - {y}"] = [round(float(np.median(err[x]) - np.median(err[y])), 4),
                               round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    del j, err, draws

T = pd.DataFrame(R)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for label in periods:
    print(f"\n== {label}: median |log err| / anomaly sign agreement / bias % ==")
    print(T[T.period == label].pivot(index="model", columns="scale",
          values=["med_abs_log_err", "anomaly_sign_agree", "bias_pct"]).round(3).to_string())
yr = T[(T.scale == "5deg") & T.period.isin(years)]
print("\n== 5deg median |log err| by year ==")
print(yr.pivot(index="period", columns="model", values="med_abs_log_err").to_string())
print("\n== bias % by year ==")
print(yr.pivot(index="period", columns="model", values="bias_pct").to_string())
print("\n== paired bootstrap, 5deg median |log err| difference [point, 2.5%, 97.5%] (negative = first is better) ==")
for k, v in boots.items():
    print(f"  {k:40s} {v}")
print(flush=True)

# fire regimes, cell-month level, 2019-2024. Bias needs only sums; the error needs pairs, and a pair
# with pred + obs >= 20 has max(pred, obs) >= 10, so only those rows are joined.
labels = np.array(["0", "<1", "1-10", "10-100", ">=100"])
reg_of = np.zeros(720 * 1440, np.int8)
reg_of[clim.index.to_numpy()] = np.digitize(clim.to_numpy(), [1e-9, 1, 10, 100])
t24 = test.loc[test.year <= 2024, ["cell", "year", "month", "V", "M"]]
obs_reg = pd.Series(t24.V.to_numpy()).groupby(reg_of[t24.cell.to_numpy()]).sum()
obs_none = t24.V[t24.M == 0].sum()
modis_keys = t24.loc[t24.M > 0, ["cell", "year", "month"]]
K = ["cell", "year", "month"]
rows = []
for k, v in models.items():
    v = v[v.year <= 2024]
    pr = pd.Series(v.pred.to_numpy(np.float64)).groupby(reg_of[v.cell.to_numpy()]).sum()
    pred_none = v.pred.to_numpy(np.float64).sum() - v.merge(modis_keys, on=K).pred.to_numpy(np.float64).sum()
    big = pd.concat([v.loc[v.pred >= 10, K], t24.loc[t24.V >= 10, K]]).drop_duplicates()
    big = big.merge(v, on=K, how="left").merge(t24, on=K, how="left").fillna(0)
    big = big[(big.pred + big.V) >= 20]
    e = np.abs(np.log((big.pred.to_numpy() + 1) / (big.V.to_numpy() + 1)))
    br = reg_of[big.cell.to_numpy()]
    for i, lab in enumerate(labels):
        rows.append(dict(model=k, regime=lab, bias_pct=round(100 * (pr.get(i, 0) / obs_reg.get(i, np.nan) - 1), 1),
                         cell_med_abs_log_err=round(float(np.median(e[br == i])), 3) if (br == i).any() else np.nan))
    none = (big.M == 0).to_numpy()
    rows.append(dict(model=k, regime="MODIS saw nothing", bias_pct=round(100 * (pred_none / obs_none - 1), 1),
                     cell_med_abs_log_err=round(float(np.median(e[none])), 3) if none.any() else np.nan))
    del big
Q = pd.DataFrame(rows)
print("\n== bias % by fire regime (MODIS dets/yr), cell-months 2019-2024 ==")
print(Q.pivot(index="regime", columns="model", values="bias_pct").to_string())
print("\n== cell-month median |log err| by fire regime (pred + obs >= 20) ==")
print(Q.pivot(index="regime", columns="model", values="cell_med_abs_log_err").to_string())

# biggest fire countries (reuses the country-scale sums from above), 2019-2024
o24 = country_obs[country_obs.index.get_level_values(1) <= 2024]
top = o24.groupby(level=0).sum().sort_values().index[-15:][::-1]
Cn = {}
for k, p in country_agg.items():
    j = pd.concat([p[p.index.get_level_values(1) <= 2024].rename("p"), o24.rename("o")], axis=1).fillna(0)
    Cn[k] = {names[c]: round(float(np.median(np.abs(np.log((j.loc[c].p + 1) / (j.loc[c].o + 1))))), 3) for c in top}
Cn = pd.DataFrame(Cn)
print("\n== top-15 fire countries: median monthly |log err| 2019-2024 ==")
print(Cn.to_string())

T.to_csv(D / "scores.csv", index=False); Q.to_csv(D / "regimes.csv", index=False)
Cn.to_csv(D / "countries.csv"); json.dump(boots, open(D / "bootstrap.json", "w"), indent=1)
print("\nwrote", D)

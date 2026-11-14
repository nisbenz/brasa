"""Is there signal left for a learned model? Explain Brasa's errors with MODIS descriptors.

Usage: python -I ml/residual_check.py <derived_dir>
No training. Fits Brasa M3 on 2012-2018, then on cell-months where MODIS saw fire:
  1. residual r = ln((V+1)/(pred+1)) on 2019-2024 (out of sample)
  2. noise floor: same quantity for NOAA-20 vs S-NPP (2019-2024): irreducible scatter
  3. per descriptor, share of residual variance explained by its decile bins
     (bin means from 2019-2021, scored on 2022-2024, so no in-sample flattery)
  4. regional test: multiply Brasa's cell predictions by the 3-descriptor bin-table correction
     learned on 2019-2021; compare 5deg and country error on 2022-2024
If (3)/(4) show little, a neural net on the same inputs is unlikely to beat Brasa.
Weights: MODIS detections, so big fire cells (which dominate regional totals) count more.
"""
import json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from harmonize import fit, ratio, predict_regions, parents

d = Path(sys.argv[1])
bad = set(json.loads((HERE.parent / "docs/validation/model.json").read_text())["excluded_training_months"])
f = duckdb.connect().sql(f"""
  SELECT cell, year, month, coalesce(m_n,0)::DOUBLE M, coalesce(v_n,0)::DOUBLE V, coalesce(n20_n,0)::DOUBLE N,
         m_terra, m_night, m_frp_sum, m_f10, m_pix, m_conf, m_days, m_maxday
  FROM '{d}/ml/features.parquet' WHERE year BETWEEN 2012 AND 2024""").df()
f = f[[f"{y}-{m:02d}" not in bad for y, m in zip(f.year, f.month)]].reset_index(drop=True)
print(f"{len(f):,} cell-months 2012-2024 (outage months removed)", flush=True)

train = f[(f.year <= 2018) & ((f.M > 0) | (f.V > 0))]
brasa = fit(train, a=20, seasonal=True, floor=True)

# neighbour-month MODIS activity in the same cell
f["t"] = f.year * 12 + f.month
mt = f.set_index(["cell", "t"]).M
f["M_adj"] = (mt.reindex(pd.MultiIndex.from_arrays([f.cell, f.t - 1])).fillna(0).to_numpy()
              + mt.reindex(pd.MultiIndex.from_arrays([f.cell, f.t + 1])).fillna(0).to_numpy())

te = f[(f.year >= 2019) & (f.M > 0)].copy()
te["pred"] = ratio(brasa, te.cell.to_numpy(), te.month.to_numpy(), te.M.to_numpy()) * te.M
te["r"] = np.log((te.V + 1) / (te.pred + 1))
te["w"] = te.M
D = {
    "MODIS count (log)": np.log(te.M),
    "night share": te.m_night / te.M,
    "Terra share": te.m_terra / te.M,
    "mean FRP per detection": te.m_frp_sum / te.M,
    "small-fire share (FRP<10)": te.m_f10 / te.M,
    "pixel footprint": te.m_pix,
    "confidence": te.m_conf,
    "fire days": te.m_days,
    "peak-day share": te.m_maxday / te.M,
    "adjacent-month share": te.M_adj / (te.M + te.M_adj),
    "calendar month": te.month,
}


def wvar(x, w):
    m = np.average(x, weights=w); return np.average((x - m) ** 2, weights=w)


A, B = (te.year <= 2021).to_numpy(), (te.year >= 2022).to_numpy()
tot = wvar(te.r[B], te.w[B])
n20 = te[(te.N > 0) | (te.V > 0)]
noise = wvar(np.log((n20.V + 1) / (n20.N + 1)), n20.w) / 2  # each instrument carries half
print(f"\nBrasa residual variance (2022-24, weighted): {tot:.3f}")
print(f"instrument noise floor (NOAA-20 vs S-NPP / 2): {noise:.3f}  -> explainable at most ~{max(tot-noise,0)/tot:.0%}")

bins = {}
rows = []
for name, x in D.items():
    x = pd.Series(np.asarray(x, float), index=te.index).fillna(-1)
    q = np.unique(np.quantile(x[A], np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(q[1:-1], x, side="right"), 0, len(q) - 2)
    bins[name] = b
    g = pd.DataFrame({"b": b[A], "r": te.r[A], "w": te.w[A]})
    mean = g.groupby("b").apply(lambda s: np.average(s.r, weights=s.w), include_groups=False)
    fit_b = pd.Series(b[B]).map(mean).fillna(0).to_numpy()
    resid = te.r[B].to_numpy() - fit_b
    rows.append((name, 1 - wvar(resid, te.w[B]) / tot))
rows.sort(key=lambda kv: -kv[1])
print("\nshare of Brasa residual variance explained out of sample (decile bin means):")
for name, v in rows:
    print(f"  {name:28s} {v:+.1%}")

# joint lookup on the top 3 descriptors -> correction factor, tested regionally
top = [n for n, _ in rows[:3]]
key = sum(bins[n] * 10 ** i for i, n in enumerate(top))
g = pd.DataFrame({"k": key[A], "r": te.r[A], "w": te.w[A]})
s = g.groupby("k").agg(rw=("r", lambda v: np.average(v, weights=g.loc[v.index, "w"])), n=("w", "sum"))
shrink = s.n / (s.n + 200)                                    # small bins pull to no correction
corr = (s.rw * shrink)
te["fac"] = np.exp(pd.Series(key).map(corr).fillna(0).to_numpy())
j = 1 - wvar(te.r[B] - np.log(te.fac[B]), te.w[B]) / tot
print(f"\njoint table on {top}: explains {j:+.1%} out of sample")

cc = pd.read_parquet(next(p for p in (d / "cell_country.parquet", d / "app/cell_country.parquet")
                          if p.exists())).set_index("cell").country
tb = f[f.year >= 2022]
mod_b = tb[tb.M > 0][["cell", "year", "month", "M"]]
teB = te[B]
print("\nregional error 2022-2024 (median |ln((p+1)/(o+1))| over region-months with p+o>=20):")
for sc, fn in {"5deg": lambda c: parents(c)[1],
               "country": lambda c: cc.reindex(c).fillna("?").to_numpy()}.items():
    base = predict_regions(brasa, mod_b, fn)
    delta = teB.assign(region=fn(teB.cell.to_numpy()), dp=teB.pred * (teB.fac - 1)) \
        .groupby(["region", "year", "month"]).dp.sum()
    obs = tb.assign(region=fn(tb.cell.to_numpy())).groupby(["region", "year", "month"]).V.sum()
    n2 = tb.assign(region=fn(tb.cell.to_numpy())).groupby(["region", "year", "month"]).N.sum()
    for label, p in (("Brasa M3", base), ("Brasa M3 x lookup", base.add(delta, fill_value=0)),
                     ("ceiling NOAA-20", n2)):
        jj = pd.concat([p.rename("p"), obs.rename("o")], axis=1).fillna(0)
        jj = jj[(jj.p + jj.o) >= 20]
        e = np.median(np.abs(np.log((jj.p + 1) / (jj.o + 1))))
        print(f"  {sc:8s} {label:20s} err {e:.3f}  bias {100 * (jj.p.sum() / jj.o.sum() - 1):+.1f}%")

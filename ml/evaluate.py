"""Score the U-Net against Brasa's hierarchical-ratio model and the NOAA-20 ceiling.

Usage: python -I ml/evaluate.py <derived_dir> <stack_dir> <model.pt>
           [--train 2012-2018] [--test 2019-2024] [--importance] [--out <dir>]
Both models predict S-NPP-equivalent counts for the test months from MODIS alone. Brasa is refit
here on the same grid, years and filters (pipeline/harmonize.fit, a=20, seasonal + floor), so the
comparison is like for like. Metrics are those of pipeline/03_validate.py, at 1deg, 5deg and
country scales, plus a cell-level negative-binomial deviance for the CNN.
--importance: permutation importance of input groups (swap a group with another test month's
values, or shift static maps spatially) on 5-degree log error.
"""
import argparse, json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pipeline"))
from core import Stack, UNet, predict_month
from harmonize import fit, predict_regions, parents


def years(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def score(pred, obs, months):
    """pipeline/03_validate.py score(), restricted to the scored (year, month) list."""
    j = pd.concat([pred.rename("p"), obs.rename("o")], axis=1).fillna(0)
    j = j[[ym in months for ym in zip(j.index.get_level_values(1), j.index.get_level_values(2))]]
    j = j[(j.p + j.o) > 0]
    big = j[(j.p + j.o) >= 20]
    le = np.log((big.p + 1) / (big.o + 1))
    k = j.reset_index()
    k = k[k.groupby("region").o.transform("sum") >= 200]
    regions = k.region.unique()
    grid = pd.MultiIndex.from_tuples([(r, y, m) for r in regions for y, m in months],
                                     names=["region", "year", "month"])
    full = k.set_index(["region", "year", "month"]).reindex(grid).fillna(0).reset_index()
    for col in ("p", "o"):
        med = full.groupby(["region", "month"])[col].transform("median")
        full[col + "_an"] = np.log((full[col] + 1) / (med + 1))
    strong = np.abs(full.o_an) > np.log(1.5)
    return dict(
        bias_pct=round(100 * (j.p.sum() / j.o.sum() - 1), 2),
        med_abs_log_err=round(float(np.median(np.abs(le))), 3) if len(le) else np.nan,
        within_25pct=round(float((np.abs(le) < np.log(1.25)).mean()), 3) if len(le) else np.nan,
        r_log=round(float(np.corrcoef(np.log1p(j.p), np.log1p(j.o))[0, 1]), 4),
        anomaly_r=round(float(np.corrcoef(full.p_an, full.o_an)[0, 1]), 4) if len(full) > 2 else np.nan,
        anomaly_sign_agree=round(float((np.sign(full.p_an) == np.sign(full.o_an))[strong].mean()), 3)
        if strong.any() else np.nan,
    )


def cnn_cells(model, s, months_idx, dev, override_fn=None):
    """Long table cell, year, month, pred for all cells with pred > 1e-3."""
    r0, c0 = s.meta["r0"], s.meta["c0"]
    out = []
    for t in months_idx:
        mu = predict_month(model, s, t, dev, override=override_fn(t) if override_fn else None)
        r, c = np.nonzero(mu > 1e-3)
        y, m = s.months[t]
        out.append(pd.DataFrame({"cell": (r + r0) * 1440 + c + c0, "year": y, "month": m,
                                 "pred": mu[r, c].astype(np.float64)}))
    return pd.concat(out, ignore_index=True)


def region_sum(df, fn, col):
    return df.assign(region=fn(df.cell.to_numpy())).groupby(["region", "year", "month"])[col].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path); ap.add_argument("stack"); ap.add_argument("model")
    ap.add_argument("--train", default="2012-2018"); ap.add_argument("--test", default="2019-2024")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--imp-months", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out or Path(a.model).with_suffix("")
    out.mkdir(parents=True, exist_ok=True)

    s = Stack(a.stack)
    meta = s.meta
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    model = UNet(ck["n_in"], ck["width"]).to(dev); model.load_state_dict(ck["state"]); model.eval()

    test_idx = s.usable(set(years(a.test)))
    test_months = [s.months[t] for t in test_idx]
    bad = set(meta["excluded_months"])
    con = duckdb.connect()
    in_grid = (f"cell // 1440 >= {meta['r0']} AND cell // 1440 < {meta['r1']} AND "
               f"cell % 1440 >= {meta['c0']} AND cell % 1440 < {meta['c1']}")
    f = con.sql(f"""SELECT cell, year, month, coalesce(m_n,0)::DOUBLE M, coalesce(v_n,0)::DOUBLE V,
                           coalesce(n20_n,0)::DOUBLE N FROM '{a.derived}/ml/features.parquet'
                    WHERE year >= 2012 AND {in_grid}""").df()
    f = f[[f"{y}-{m:02d}" not in bad for y, m in zip(f.year, f.month)]]
    tr_years, te_years = set(years(a.train)), set(years(a.test))
    train = f[f.year.isin(tr_years) & ((f.M > 0) | (f.V > 0))]
    test = f[f.year.isin(te_years)]

    cc_path = next(p for p in (a.derived / "cell_country.parquet", a.derived / "app" / "cell_country.parquet")
                   if p.exists())
    cc = pd.read_parquet(cc_path).set_index("cell").country
    scales = {"1deg": lambda c: parents(c)[0], "5deg": lambda c: parents(c)[1],
              "country": lambda c: cc.reindex(c).fillna("?").to_numpy()}

    print(f"test months {len(test_idx)}; predicting with CNN on {dev} ...", flush=True)
    cells = cnn_cells(model, s, test_idx, dev)
    brasa = fit(train, a=20, seasonal=True, floor=True)
    modis_test = test[test.M > 0][["cell", "year", "month", "M"]]

    rows = {}
    for sc, fn in scales.items():
        obs = region_sum(test, fn, "V")
        rows[("Brasa M3 (refit)", sc)] = score(predict_regions(brasa, modis_test, fn), obs, test_months)
        rows[("U-Net", sc)] = score(region_sum(cells, fn, "pred"), obs, test_months)
        n20m = [ym for ym in test_months if ym[0] >= 2018]
        if n20m:
            rows[("CEILING NOAA-20 vs S-NPP", sc)] = score(region_sum(test, fn, "N"), obs, n20m)
        for k in [k for k in rows if k[1] == sc]:
            print(k, rows[k], flush=True)
    table = pd.DataFrame(rows).T
    table.index.names = ["model", "scale"]
    table.to_csv(out / "comparison.csv")
    print(table.to_string())

    if a.importance:
        rng = np.random.default_rng(0)
        idx = list(rng.choice(test_idx, size=min(a.imp_months, len(test_idx)), replace=False))
        ym = [s.months[t] for t in idx]
        obs5 = region_sum(test, scales["5deg"], "V")

        def err(cells_):
            return score(region_sum(cells_, scales["5deg"], "pred"), obs5, ym)["med_abs_log_err"]

        base = err(cnn_cells(model, s, idx, dev))
        ch = meta["channels"]
        groups = {"count (m_n, m_n_all)": ["m_n", "m_n_all"],
                  "FRP histogram": ["m_f10", "m_f30", "m_f100", "m_fbig"],
                  "FRP total/max": ["m_frp_sum", "m_frp_max"],
                  "Terra / night share": ["m_terra", "m_night"],
                  "fire days / burstiness": ["m_days", "m_maxday"],
                  "pixel footprint": ["m_pix"], "confidence / brightness": ["m_conf", "m_bright"]}
        imp = {}
        orig = s.inputs
        for name, cols in groups.items():
            ci = [ch.index(c) for c in cols]
            other = {t: rng.choice([u for u in test_idx if u != t]) for t in idx}

            def patched(t, win=None, override=None, ci=ci):
                x = orig(t, win, override)
                xo = orig(other[t], win, override)
                for k in ci:                       # all three month blocks
                    for b in range(3):
                        x[b * s.C + k] = xo[b * s.C + k]
                return x
            s.inputs = patched
            imp[name] = err(cnn_cells(model, s, idx, dev)) - base
            s.inputs = orig
        swap = {t: rng.choice([u for u in test_idx if u != t]) for t in idx}
        imp["neighbour months (t-1, t+1)"] = err(cnn_cells(
            model, s, idx, dev, lambda t: {"prev": swap[t], "next": swap[t]})) - base
        for name, sl in (("location (lat/lon)", slice(3 * s.C, 3 * s.C + 3)),
                         ("MODIS climatology", slice(3 * s.C + 3, 3 * s.C + 5))):
            def shifted(t, win=None, override=None, sl=sl):
                x = orig(t, win, override)
                x[sl] = np.roll(x[sl], (x.shape[1] // 3, x.shape[2] // 3), axis=(1, 2))
                return x
            s.inputs = shifted
            imp[name] = err(cnn_cells(model, s, idx, dev)) - base
            s.inputs = orig
        imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        json.dump(dict(baseline_5deg_med_abs_log_err=base, increase=imp), open(out / "importance.json", "w"),
                  indent=1)
        print("permutation importance (increase in 5deg median |log err|):")
        for k, v in imp.items():
            print(f"  {k:32s} {v:+.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()

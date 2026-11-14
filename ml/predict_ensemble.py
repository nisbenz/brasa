"""Run the U-Net + MLP ensemble: VIIRS S-NPP-equivalent counts per cell-month from MODIS alone.

Usage: python -I ml/predict_ensemble.py <stack_dir> <unet.pt> <ctx_models.pt> <out_dir>
           [--start 2002-07] [--end 2025-12] [--countries <cell_country.parquet>]
The stack must be built over the wanted years (ml/build_tensor.py --years 2002-2025 --name full);
VIIRS is not needed, so pre-2012 months work. The default start skips the Terra-only months
(Aqua MODIS from July 2002): the models were trained on Terra + Aqua and would under-predict there.
Writes <out_dir>/cells_<year>.parquet (cell, year, month, unet, mlp, pred = mean of the two) and
<out_dir>/monthly_totals.csv (MODIS count, the three predictions and observed S-NPP where it exists,
globally and per country if --countries is given).
On 2019-2024 the ensemble scores 5deg median |log err| 0.203 vs 0.246 for Brasa M3 (ml/compare.py).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from core import Stack, UNet, context_features, context_net, predict_month

ap = argparse.ArgumentParser()
ap.add_argument("stack"); ap.add_argument("unet"); ap.add_argument("ctx"); ap.add_argument("out", type=Path)
ap.add_argument("--start", default="2002-07"); ap.add_argument("--end", default="2025-12")
ap.add_argument("--countries", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
s = Stack(a.stack); H, W = s.shape; r0, c0 = s.meta["r0"], s.meta["c0"]
ck = torch.load(a.unet, map_location=dev, weights_only=False)
unet = UNet(ck["n_in"], ck["width"]).to(dev); unet.load_state_dict(ck["state"]); unet.eval()
cx = torch.load(a.ctx, map_location=dev, weights_only=False)
mlp = context_net("mlp", len(cx["mu"])).to(dev); mlp.load_state_dict(cx["mlp"]); mlp.eval()
mu_, sd_ = cx["mu"].to(dev), cx["sd"].to(dev)
ym = lambda s_: tuple(map(int, s_.split("-")))
months = [t for t, m in enumerate(s.months) if ym(a.start) <= m <= ym(a.end)]
if not months:
    raise SystemExit(f"no months in {a.start}..{a.end}; stack has {s.meta['months'][0]}..{s.meta['months'][-1]}")
code = names = None
if a.countries:
    cc = pd.read_parquet(a.countries)
    names = np.array(sorted(cc.country.dropna().unique().tolist()) + ["?"], dtype=object)
    code = np.full(720 * 1440, len(names) - 1, np.int16)
    code[cc.cell.to_numpy()] = pd.Categorical(cc.country.fillna("?"), categories=names).codes
a.out.mkdir(parents=True, exist_ok=True)
cells_all = (np.arange(r0, r0 + H)[:, None] * 1440 + np.arange(c0, c0 + W)[None]).reshape(-1)
t0, year_rows, totals = time.time(), [], []
with torch.no_grad():
    for k, t in enumerate(months):
        y, mo = s.months[t]
        u = predict_month(unet, s, t, dev).reshape(-1)
        f, b = context_features(s, t)
        g = mlp(((torch.from_numpy(f).to(dev) - mu_) / sd_))[:, 0]
        m = torch.exp((torch.from_numpy(b).to(dev) + g).clamp(-12, 12)).cpu().numpy()
        p = 0.5 * (u + m)
        M = np.expm1(s.arr["X"][t, 0].astype(np.float32)).reshape(-1)
        V = s.target(t).reshape(-1)
        keep = np.nonzero(np.maximum(u, m) > 1e-3)[0]
        year_rows.append(pd.DataFrame({"cell": cells_all[keep].astype(np.int32), "year": np.int16(y),
                                       "month": np.int8(mo), "unet": u[keep].astype(np.float32),
                                       "mlp": m[keep].astype(np.float32), "pred": p[keep].astype(np.float32)}))
        obs = float(np.nansum(V)) if np.isfinite(V).any() else np.nan
        totals.append(dict(region="GLOBAL", year=y, month=mo, modis=float(M.sum()), unet=float(u.sum()),
                           mlp=float(m.sum()), pred=float(p.sum()), viirs_obs=obs))
        if code is not None:
            reg = code[cells_all]
            df = pd.DataFrame({"r": reg, "modis": M, "unet": u, "mlp": m, "pred": p,
                               "viirs_obs": V}).groupby("r").sum(min_count=1)
            for r_, row in df.iterrows():
                totals.append(dict(region=names[r_], year=y, month=mo, **row.to_dict()))
        last = k + 1 == len(months) or s.months[months[k + 1]][0] != y
        if last:
            pd.concat(year_rows, ignore_index=True).to_parquet(a.out / f"cells_{y}.parquet")
            year_rows = []
            print(f"{y} done ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(totals).to_csv(a.out / "monthly_totals.csv", index=False)
print(f"wrote {a.out}: {len(months)} months, {a.start}..{a.end}")

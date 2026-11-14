"""Per-pixel baselines with hand-built neighbourhood features: is the U-Net's gain just pooling?

Usage: python -I ml/context_baseline.py <stack_dir> <out_dir> [--test 2019-2025] [--no-predict]
Same residual form as the U-Net, log mu = log(4.1 M + 0.05) + g(features), Poisson loss, but no
learned convolutions (features: core.context_features). g is linear or a 2x128 MLP.
Train 2012-2017, early-stop on 2018. Writes <out_dir>/ctx_models.pt (both nets + feature
normalization, used by ml/predict_ensemble.py) and, unless --no-predict, ctx_linear_cells.parquet
and ctx_mlp_cells.parquet for the test years, for ml/compare.py.
Result on 2019-2024 (5deg median |log err|): linear 3.09 (extrapolates badly into empty cells;
ignore), MLP 0.245, U-Net 0.211, mean of U-Net and MLP 0.203, Brasa M3 0.246.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from core import Stack, context_features, context_net

ap = argparse.ArgumentParser()
ap.add_argument("stack"); ap.add_argument("out", type=Path)
ap.add_argument("--test", default="2019-2025")
ap.add_argument("--no-predict", action="store_true", help="only train and save ctx_models.pt")
ap.add_argument("--frac-empty", type=float, default=0.03, help="share of empty pixels kept")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
s = Stack(a.stack); H, W = s.shape; r0, c0 = s.meta["r0"], s.meta["c0"]


def sample(months, rng):
    Fs, Bs, Ys, Ws = [], [], [], []
    for t in months:
        f, b = context_features(s, t)
        y = s.target(t).reshape(-1)
        busy = (f[:, 3] + f[:, 7] + f[:, 11] > 0) | (y > 0)   # MODIS within 25 cells in t-1..t+1, or VIIRS
        keep = np.isfinite(y) & (busy | (rng.random(len(y)) < a.frac_empty))
        Fs.append(f[keep]); Bs.append(b[keep]); Ys.append(y[keep])
        Ws.append(np.where(busy, 1.0, 1 / a.frac_empty)[keep].astype(np.float32))
    return [torch.from_numpy(np.concatenate(v)) for v in (Fs, Bs, Ys, Ws)]


def pois(g, b, y, w):
    lm = (b + g).clamp(-12, 12)
    return (w * (torch.exp(lm) - y * lm)).sum() / w.sum()


def train(kind, tr, va):
    net = context_net(kind, tr[0].shape[1])
    last = net if kind == "linear" else net[-1]
    torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-2 if kind == "linear" else 3e-3)
    best, state, bad = float("inf"), None, 0
    for ep in range(30):
        net.train()
        for j in torch.randperm(len(tr[2]), device=dev).split(16384):
            loss = pois(net(tr[0][j])[:, 0], tr[1][j], tr[2][j], tr[3][j])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = sum(float(pois(net(va[0][j])[:, 0], va[1][j], va[2][j], va[3][j])) * len(j)
                    for j in torch.arange(len(va[2]), device=dev).split(1 << 18)) / len(va[2])
        print(f"{kind} epoch {ep} val {v:.5f}", flush=True)
        if v < best - 1e-5:
            best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(state)
    return net


t0 = time.time()
rng = np.random.default_rng(0)
tr = sample(s.usable(set(range(2012, 2018))), rng)
va = sample(s.usable({2018}), rng)
mu_, sd_ = tr[0].mean(0), tr[0].std(0) + 1e-6
tr[0].sub_(mu_).div_(sd_); va[0].sub_(mu_).div_(sd_)   # in place: ~3 GB each
tr, va = [x.to(dev) for x in tr], [x.to(dev) for x in va]
print(f"train rows {len(tr[2]):,} val rows {len(va[2]):,} features {tr[0].shape[1]} "
      f"device {dev} ({time.time() - t0:.0f}s)", flush=True)
nets = {k: train(k, tr, va) for k in ("linear", "mlp")}
del tr, va
a.out.mkdir(parents=True, exist_ok=True)
torch.save({"mu": mu_, "sd": sd_, **{k: n.cpu().state_dict() for k, n in nets.items()}}, a.out / "ctx_models.pt")
print("saved", a.out / "ctx_models.pt", flush=True)
if a.no_predict:
    sys.exit()

y0, _, y1 = a.test.partition("-")
rows = {k: [] for k in nets}
with torch.no_grad():
    for t in s.usable(set(range(int(y0), int(y1 or y0) + 1))):
        f, b = context_features(s, t)
        f = ((torch.from_numpy(f) - mu_) / sd_).to(dev); b = torch.from_numpy(b).to(dev)
        y, mo = s.months[t]
        for k, net in nets.items():
            net.to(dev)
            mu = torch.exp((b + net(f)[:, 0]).clamp(-12, 12)).cpu().numpy().reshape(H, W)
            r, c = np.nonzero(mu > 1e-3)
            rows[k].append(pd.DataFrame({"cell": (r + r0) * 1440 + c + c0, "year": y, "month": mo,
                                         "pred": mu[r, c].astype(np.float64)}))
for k, v in rows.items():
    pd.concat(v, ignore_index=True).to_parquet(a.out / f"ctx_{k}_cells.parquet")
print(f"wrote {a.out} ({time.time() - t0:.0f}s)")

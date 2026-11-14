"""Per-cell U-Net predictions for a range of months, saved for ml/compare.py.

Usage: python -I ml/predict_cells.py <stack_dir> <model.pt> <out.parquet> [--years 2019-2025]
Writes cell, year, month, pred for every cell with pred > 1e-3.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from core import Stack, UNet, predict_month

ap = argparse.ArgumentParser()
ap.add_argument("stack"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--years", default="2019-2025")
a = ap.parse_args()
y0, _, y1 = a.years.partition("-")
dev = "cuda" if torch.cuda.is_available() else "cpu"
s = Stack(a.stack)
ck = torch.load(a.model, map_location=dev, weights_only=False)
model = UNet(ck["n_in"], ck["width"]).to(dev); model.load_state_dict(ck["state"]); model.eval()
r0, c0 = s.meta["r0"], s.meta["c0"]
t0, rows = time.time(), []
for t in s.usable(set(range(int(y0), int(y1 or y0) + 1))):
    mu = predict_month(model, s, t, dev)
    r, c = np.nonzero(mu > 1e-3)
    y, m = s.months[t]
    rows.append(pd.DataFrame({"cell": (r + r0) * 1440 + c + c0, "year": y, "month": m,
                              "pred": mu[r, c].astype(np.float64)}))
pd.concat(rows, ignore_index=True).to_parquet(a.out)
print(f"{len(rows)} months -> {a.out} ({time.time() - t0:.0f}s)")

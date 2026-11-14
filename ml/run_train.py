"""Step 3 - training: MODIS -> VIIRS-equivalent U-Net (or an ensemble of them) on random patches.

Usage: python -I ml/run_train.py <stack_dir> <out.pt> [options]       (--help lists all options)
       python -I ml/run_train.py <stack_dir> <out.pt> --smoke         # tiny CPU run, checks plumbing

Members and selection
  --train 2012-2017 --val 2018    one member per seed, early-stopped on the val years
  --train 2012-2018 --folds 7     blocked CV: member k validates on year block k, trains on the rest;
                                  the .pt holds all members and predictions average them
  --train 2012-2025 --val none --epochs N
                                  final refit on every overlap year with a fixed epoch count
                                  (take N from the best epochs of a CV run, printed at the end)
  --seeds S                       S members per fold (different init and patch order)
Validation every --val-every epochs: NB loss on fixed val patches, and the 5-degree median
|log err| of full-map predictions of up to --val-months val months (what the product is scored on).
--select metric|nll picks the checkpoint on one of them; --patience stops early.

What changed vs the legacy ml/train.py (all switchable for ablations)
  --preset v2       per-satellite Terra / Aqua channels + Aqua flag (legacy: 15 combined channels)
  --sampling input  patches drawn in proportion to sqrt(MODIS count), not sqrt(VIIRS count):
                    choosing samples by the target biases E[y | x]; choosing by the input does not
  --aqua-dropout p  zero the Aqua block (and flag) on a share p of patches, so one model also covers
                    the Terra-only months 2000-11 .. 2002-06
  --reg-scales      regional loss over 1/2/5-degree blocks (legacy: 2 degrees only)
  --norm layer      per-pixel LayerNorm: a cell's output no longer depends on the tile it sits in
                    (GroupNorm made patch training and full-map inference disagree)
  --depth / --no-location / --offsets   receptive field, lat/lon inputs, temporal context
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import (LOCATION_STATIC, PRESETS, STATIC_NAMES, InputSpec, ModelConfig,  # noqa: E402
                        TrainConfig, parse_years)
from lib_metrics import Regions, map_score  # noqa: E402
from lib_model import UNet, nb_nll, pick_device, regional_log_loss, save_checkpoint, tiled_forward  # noqa: E402
from lib_stack import Stack  # noqa: E402


class Patches(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """P x P patches drawn with probability ~ sqrt(patch count) + 5% of the mean (so empty land
    still appears and teaches the model where *not* to predict fire). The count is MODIS (input)
    or VIIRS (target, legacy behaviour)."""

    def __init__(self, stack: Stack, spec: InputSpec, months: list[int], P: int, n: int,
                 sampling: str = "input", aqua_dropout: float = 0.0, seed: int = 0,
                 fixed: bool = False) -> None:
        self.s, self.spec, self.P, self.n, self.fixed, self.seed = stack, spec, P, n, fixed, seed
        self.aqua_dropout = aqua_dropout if any(c.startswith("a_") for c in spec.channels) else 0.0
        H, W = stack.shape
        st = max(P // 2, 1)
        rows = np.arange(0, max(H - P, 0) + 1, st)
        cols = np.arange(0, max(W - P, 0) + 1, st)
        cand: list[tuple[int, int, int]] = []
        wts: list[float] = []
        for t in months:
            m = np.nan_to_num(stack.target(t) if sampling == "target" else stack.modis_count(t))
            ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))        # integral image
            for r in rows:
                for c in cols:
                    r1, c1 = min(r + P, H), min(c + P, W)
                    cand.append((t, int(r), int(c)))
                    wts.append(float(ii[r1, c1] - ii[r, c1] - ii[r1, c] + ii[r, c]))
        w = np.sqrt(np.maximum(np.array(wts), 0))
        w += 0.05 * w.mean() + 1e-9
        self.cand = np.array(cand)
        self.p = w / w.sum()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + i if self.fixed else None)
        t, r, c = (int(v) for v in self.cand[rng.choice(len(self.cand), p=self.p)])
        H, W = self.s.shape
        if not self.fixed:   # jitter inside the stride
            r = int(np.clip(r + rng.integers(-self.P // 4, self.P // 4 + 1), 0, max(H - self.P, 0)))
            c = int(np.clip(c + rng.integers(-self.P // 4, self.P // 4 + 1), 0, max(W - self.P, 0)))
        win = (r, min(r + self.P, H), c, min(c + self.P, W))
        off = bool(self.aqua_dropout) and self.s.has_aqua(t) and rng.random() < self.aqua_dropout
        x = self.s.inputs(t, self.spec, win, aqua_off=off)
        return torch.from_numpy(x), torch.from_numpy(self.s.target(t, win))


def step_loss(model: UNet, x: torch.Tensor, y: torch.Tensor, reg_w: float, scales: tuple[int, ...]
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lm, la = model(x)
    valid = torch.isfinite(y)
    y0 = torch.nan_to_num(y)
    pix = nb_nll(y0, lm, la)[valid].mean()
    reg = regional_log_loss(y0 * valid, torch.exp(lm) * valid, scales)
    return pix + reg_w * reg, pix.detach(), reg.detach()


def spread(months: list[int], k: int) -> list[int]:
    """Up to k months evenly spaced through the list (keeps every season in validation)."""
    if len(months) <= k:
        return months
    return [months[int(i)] for i in np.linspace(0, len(months) - 1, k).round()]


def train_member(stack: Stack, cfg: ModelConfig, tc: TrainConfig, train_m: list[int], val_m: list[int],
                 seed: int, dev: str, label: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    spec = cfg.spec
    dl = torch.utils.data.DataLoader(
        Patches(stack, spec, train_m, tc.patch, tc.steps * tc.batch, tc.sampling, tc.aqua_dropout, seed),
        batch_size=tc.batch, num_workers=tc.workers, pin_memory=dev == "cuda",
        persistent_workers=tc.workers > 0)
    vdl = torch.utils.data.DataLoader(
        Patches(stack, spec, val_m, tc.patch, tc.val_patches, tc.sampling, 0.0, 123, fixed=True),
        batch_size=tc.batch, num_workers=tc.workers) if val_m else None
    regions = Regions.build(stack.cells(), scales=("5deg",)) if val_m else None
    full_m = spread(val_m, tc.val_months)
    model = UNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    total = tc.epochs * tc.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(total // 20, 1)) * 0.5 * (1 + math.cos(math.pi * k / total)))
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    history: list[dict[str, float]] = []
    for ep in range(tc.epochs):
        model.train()
        t0, acc = time.time(), []
        for x, y in dl:
            loss, pix, reg = step_loss(model, x.to(dev), y.to(dev), tc.reg_w, tc.reg_scales)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            acc.append([loss.item(), pix.item(), reg.item()])
        row: dict[str, float] = {"epoch": ep, "train": float(np.mean([a[0] for a in acc])),
                                 "train_nb": float(np.mean([a[1] for a in acc])),
                                 "train_reg": float(np.mean([a[2] for a in acc]))}
        if vdl is not None and (ep % tc.val_every == 0 or ep == tc.epochs - 1):
            model.eval()
            with torch.no_grad():
                v = np.mean([[t.item() for t in step_loss(model, x.to(dev), y.to(dev), tc.reg_w,
                                                          tc.reg_scales)] for x, y in vdl], 0)
            row.update(val=float(v[0]), val_nb=float(v[1]), val_reg=float(v[2]))
            sc = map_score(lambda t: np.exp(tiled_forward(model, stack.inputs(t, spec), dev)[0]),
                           stack.target, full_m, [stack.months[t][1] for t in full_m], regions)  # type: ignore[arg-type]
            row.update(val_5deg_err=sc["med_abs_log_err"], val_5deg_bias=sc["bias_pct"])
            crit = row["val_5deg_err"] if tc.select == "metric" else row["val"]
            if crit < best:
                best, best_ep, bad = crit, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(row)
        print(f"[{label}] " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                       for k, v in row.items()) + f" {time.time() - t0:.0f}s", flush=True)
        if vdl is not None and bad >= tc.patience:
            print(f"[{label}] early stop at epoch {ep} (best {best_ep})", flush=True)
            break
    if best_state is None:                     # no validation: keep the last epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_ep = len(history) - 1
    return {"state": best_state, "seed": seed, "epoch": best_ep, "best": best,
            "val_years": sorted({stack.months[t][0] for t in val_m}),
            "train_years": sorted({stack.months[t][0] for t in train_m}), "history": history}


def folds_of(years: list[int], k: int) -> list[list[int]]:
    """k contiguous year blocks (blocked CV keeps neighbouring, correlated years together)."""
    return [[int(y) for y in b] for b in np.array_split(np.array(years), k) if len(b)]


def build_spec(a: argparse.Namespace, stack: Stack) -> InputSpec:
    channels = PRESETS[a.preset] + (tuple(c for c in stack.channels if c.startswith("aux_")) if a.aux else ())
    per_sat = not any(c.startswith("m_") for c in channels)
    static = tuple(s for s in STATIC_NAMES if not (a.no_location and s in LOCATION_STATIC))
    if a.aux:
        static += tuple(c for c in stack.static_names if c.startswith("aux_"))
    return InputSpec(channels=channels, offsets=tuple(int(o) for o in a.offsets.split(",")),
                     static=static, clim_month=not a.no_clim, season=True, aqua_flag=per_sat,
                     base_channels=("t_n", "a_n") if per_sat else ("m_n",))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack")
    ap.add_argument("out")
    ap.add_argument("--train", default="2012-2017")
    ap.add_argument("--val", default="2018", help="'none' for a fixed-epoch refit")
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v2")
    ap.add_argument("--offsets", default="-1,0,1", help="months of MODIS context, e.g. -12,-1,0,1")
    ap.add_argument("--no-location", action="store_true", help="drop lat / lon inputs")
    ap.add_argument("--no-clim", action="store_true", help="drop the calendar-month climatology")
    ap.add_argument("--aux", action="store_true", help="also read every aux_* channel of the stack")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--norm", choices=["layer", "group"], default="layer",
                    help="layer: tile-independent (default); group: legacy GroupNorm")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=500, help="steps per epoch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg-w", type=float, default=0.3)
    ap.add_argument("--reg-scales", default="4,8,20")
    ap.add_argument("--sampling", choices=["input", "target"], default="input")
    ap.add_argument("--aqua-dropout", type=float, default=0.15)
    ap.add_argument("--val-patches", type=int, default=512)
    ap.add_argument("--val-months", type=int, default=12)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select", choices=["metric", "nll"], default="metric")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.width, a.patch, a.batch, a.epochs, a.steps, a.val_patches, a.workers, a.val_months = \
            8, 32, 4, 2, 10, 8, 0, 2
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = pick_device()

    s = Stack(a.stack)
    spec = build_spec(a, s)
    cfg = ModelConfig(spec=spec, width=a.width, depth=a.depth, norm=a.norm)
    if a.patch % 2 ** a.depth:
        raise SystemExit(f"--patch must be a multiple of 2^depth = {2 ** a.depth}")
    tc = TrainConfig(train_years=parse_years(a.train), val_years=parse_years(a.val), folds=a.folds,
                     seeds=a.seeds, patch=a.patch, batch=a.batch, epochs=a.epochs, steps=a.steps, lr=a.lr,
                     reg_w=a.reg_w, reg_scales=tuple(int(k) for k in a.reg_scales.split(",")),
                     sampling=a.sampling, aqua_dropout=a.aqua_dropout if spec.aqua_flag else 0.0,
                     val_patches=a.val_patches, val_months=a.val_months, val_every=a.val_every,
                     patience=a.patience, select=a.select, workers=a.workers, seed=a.seed)
    s.check(spec)

    if tc.folds > 1:
        plan = [(sorted(set(tc.train_years) - set(b)), b) for b in folds_of(tc.train_years, tc.folds)]
    else:
        plan = [(tc.train_years, tc.val_years)]
    print(f"device {dev} | grid {s.shape} | inputs {spec.n_in} ({a.preset}, offsets {spec.offsets}) | "
          f"depth {cfg.depth} (radius ~{cfg.receptive_radius()} cells) | "
          f"{len(plan)} fold(s) x {tc.seeds} seed(s)", flush=True)
    print(f"parameters per member {sum(p.numel() for p in UNet(cfg).parameters()):,}", flush=True)
    members: list[dict[str, Any]] = []
    for f, (tr_y, va_y) in enumerate(plan):
        tr, va = s.usable(tr_y), s.usable(va_y)
        if not tr:
            raise SystemExit(f"no usable training months in {tr_y}")
        for k in range(tc.seeds):
            label = f"fold {f} seed {k}"
            print(f"[{label}] train {tr_y[0]}-{tr_y[-1]} ({len(tr)} months) val {va_y or 'none'}", flush=True)
            members.append(train_member(s, cfg, tc, tr, va, tc.seed + 1000 * f + k, dev, label))
    info = {"train_years": sorted(set(tc.train_years)), "val_years": sorted(set(tc.val_years)),
            "train_config": asdict(tc), "stack": str(a.stack), "stack_preset": s.meta.get("preset"),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(a.out, cfg, members, info)
    best_eps = [m["epoch"] for m in members]
    print(json.dumps({"members": len(members), "best_epochs": best_eps,
                      "median_best_epoch": float(np.median(best_eps)),
                      "best": [round(float(m["best"]), 4) for m in members]}), flush=True)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

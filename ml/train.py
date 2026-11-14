"""Train the MODIS -> VIIRS-equivalent U-Net on random patches of the overlap period.

Usage: python -I ml/train.py <stack_dir> <out.pt> [--train 2012-2017] [--val 2018] ...
       python -I ml/train.py <stack_dir> <out.pt> --smoke     # tiny CPU run: checks the plumbing
Default split mirrors pipeline/03_validate.py: fit on <= 2018 (2018 used for early stopping),
test 2019-2024 with ml/evaluate.py.
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from core import Stack, UNet, nb_nll, regional_log_loss


def years(s):
    a, _, b = s.partition("-")
    return set(range(int(a), int(b or a) + 1))


class Patches(torch.utils.data.Dataset):
    """Random P x P patches, sampled in proportion to sqrt(VIIRS count) so fire regions dominate
    but empty land still shows up (it teaches the model where *not* to predict fire)."""

    def __init__(self, stack, months, P, n, seed=0, fixed=False):
        self.s, self.P, self.n, self.fixed, self.seed = stack, P, n, fixed, seed
        H, W = stack.shape
        st = max(P // 2, 1)
        rows, cols = np.arange(0, max(H - P, 0) + 1, st), np.arange(0, max(W - P, 0) + 1, st)
        cand, wts = [], []
        for t in months:
            y = np.nan_to_num(stack.target(t))
            for r in rows:
                for c in cols:
                    cand.append((t, r, c)); wts.append(y[r:r + P, c:c + P].sum())
        wts = np.sqrt(np.array(wts, float))
        wts += 0.05 * wts.mean() + 1e-9
        self.cand, self.p = np.array(cand), wts / wts.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i if self.fixed else None)
        t, r, c = self.cand[rng.choice(len(self.cand), p=self.p)]
        H, W = self.s.shape
        if not self.fixed:  # jitter inside the stride
            r = int(np.clip(r + rng.integers(-self.P // 4, self.P // 4 + 1), 0, max(H - self.P, 0)))
            c = int(np.clip(c + rng.integers(-self.P // 4, self.P // 4 + 1), 0, max(W - self.P, 0)))
        win = (r, min(r + self.P, H), c, min(c + self.P, W))
        return torch.from_numpy(self.s.inputs(t, win)), torch.from_numpy(self.s.target(t, win))


def step_loss(model, x, y, reg_w):
    lm, la = model(x)
    valid = torch.isfinite(y)
    y0 = torch.nan_to_num(y)
    pix = nb_nll(y0, lm, la)[valid].mean()
    reg = regional_log_loss(y0 * valid, torch.exp(lm) * valid)
    return pix + reg_w * reg, pix.detach(), reg.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stack"); ap.add_argument("out")
    ap.add_argument("--train", default="2012-2017"); ap.add_argument("--val", default="2018")
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=500, help="steps per epoch")
    ap.add_argument("--val-patches", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--reg-w", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.width, a.patch, a.batch, a.epochs, a.steps, a.val_patches, a.workers = 8, 32, 4, 2, 10, 8, 0
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    s = Stack(a.stack)
    tr, va = s.usable(years(a.train)), s.usable(years(a.val))
    if not tr or not va:
        raise SystemExit(f"no usable months: train={len(tr)} val={len(va)}")
    print(f"device {dev} | grid {s.shape} | train months {len(tr)} | val months {len(va)} | inputs {s.n_in}")
    dl = torch.utils.data.DataLoader(Patches(s, tr, a.patch, a.steps * a.batch), batch_size=a.batch,
                                     num_workers=a.workers, pin_memory=dev == "cuda",
                                     persistent_workers=a.workers > 0)
    vdl = torch.utils.data.DataLoader(Patches(s, va, a.patch, a.val_patches, seed=123, fixed=True),
                                      batch_size=a.batch, num_workers=a.workers)
    model = UNet(s.n_in, a.width).to(dev)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(total // 20, 1)) * 0.5 * (1 + math.cos(math.pi * k / total)))
    best = float("inf")
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); acc = []
        for x, y in dl:
            loss, pix, reg = step_loss(model, x.to(dev), y.to(dev), a.reg_w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); acc.append([loss.item(), pix.item(), reg.item()])
        model.eval(); vacc = []
        with torch.no_grad():
            for x, y in vdl:
                vacc.append([v.item() for v in step_loss(model, x.to(dev), y.to(dev), a.reg_w)])
        tl, vl = np.mean(acc, 0), np.mean(vacc, 0)
        print(f"epoch {ep:3d} train {tl[0]:.4f} (nb {tl[1]:.4f} reg {tl[2]:.4f}) "
              f"val {vl[0]:.4f} (nb {vl[1]:.4f} reg {vl[2]:.4f}) {time.time() - t0:.0f}s", flush=True)
        if vl[0] < best:
            best = vl[0]
            torch.save(dict(state=model.state_dict(), width=a.width, n_in=s.n_in, args=vars(a),
                            epoch=ep, val=float(vl[0])), a.out)
    print(f"best val {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()

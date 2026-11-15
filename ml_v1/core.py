"""Shared pieces: map-stack access, the U-Net, the count likelihood, tiled inference.

Input for target month t (per pixel, 3*C + 7 channels):
  X[t-1], X[t], X[t+1]           MODIS channels for the month and its neighbours
                                 (fires MODIS saw starting/ending in adjacent months)
  S                              lat, sin/cos lon, MODIS annual climatology
  CL[month]                      MODIS climatology of this calendar month
  sin/cos(month)                 season
Output per pixel: log mu (expected VIIRS S-NPP count) and log alpha (NB overdispersion).
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stack:
    """Lazy memmap view of one build_tensor.py output directory (safe to pickle to workers)."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.months = [tuple(map(int, s.split("-"))) for s in self.meta["months"]]
        self.ordinal = np.array([y * 12 + m - 1 for y, m in self.months])
        self.C = len(self.meta["channels"])
        self.n_in = 3 * self.C + len(self.meta["static"]) + 1 + 2
        self._arr = None

    def __getstate__(self):
        s = self.__dict__.copy(); s["_arr"] = None; return s

    @property
    def arr(self):
        if self._arr is None:
            self._arr = {k: np.load(self.path / f"{k}.npy", mmap_mode="r")
                         for k in ("X", "Y", "N20", "S", "CL")}
        return self._arr

    @property
    def shape(self):
        return self.arr["Y"].shape[1:]

    def _neighbour(self, t, dt):
        u = t + dt
        if 0 <= u < len(self.months) and self.ordinal[u] == self.ordinal[t] + dt:
            return u
        return None

    def inputs(self, t, win=None, override=None):
        """(n_in, h, w) float32 inputs for month index t over window (r0, r1, c0, c1).
        override: optional dict {"prev"|"cur"|"next": month index} used by permutation tests."""
        a = self.arr
        r0, r1, c0, c1 = win or (0, self.shape[0], 0, self.shape[1])
        h, w = r1 - r0, c1 - c0
        override = override or {}
        parts = []
        for key, dt in (("prev", -1), ("cur", 0), ("next", 1)):
            u = override.get(key, self._neighbour(t, dt) if dt else t)
            parts.append(np.zeros((self.C, h, w), np.float32) if u is None
                         else a["X"][u, :, r0:r1, c0:c1].astype(np.float32))
        m = self.months[t][1]
        parts.append(np.array(a["S"][:, r0:r1, c0:c1], np.float32))
        parts.append(np.array(a["CL"][m - 1, r0:r1, c0:c1], np.float32)[None])
        ang = 2 * np.pi * (m - 0.5) / 12
        parts.append(np.broadcast_to(np.array([np.sin(ang), np.cos(ang)], np.float32)[:, None, None],
                                     (2, h, w)))
        return np.concatenate(parts, 0)

    def target(self, t, win=None):
        r0, r1, c0, c1 = win or (0, self.shape[0], 0, self.shape[1])
        return np.array(self.arr["Y"][t, r0:r1, c0:c1], np.float32)

    def usable(self, years):
        """Month indices in `years` with S-NPP data, excluding known outage months."""
        bad = set(self.meta["excluded_months"])
        return [i for i, (y, m) in enumerate(self.months)
                if y in years and f"{y}-{m:02d}" not in bad and y >= 2012]


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.GELU())


class UNet(nn.Module):
    """3-level U-Net; receptive field ~ 70 cells (~17 degrees) at the bottleneck.
    Residual on the global-ratio baseline: log mu = log(K0 * M_t + B0) + f(x), where M_t is this
    month's MODIS count. f starts at 0, so training begins from Brasa's M0 model and learns only
    the correction; exp(f) can be mapped directly as a learned conversion-factor field."""
    K0, B0 = 4.1, 0.05

    def __init__(self, n_in, width=32):
        super().__init__()
        self.m_idx = (n_in - 7) // 3   # m_n of the current-month block (see Stack.inputs)
        w = [width, 2 * width, 4 * width, 8 * width]
        self.stem = nn.Conv2d(n_in, w[0], 1)        # per-pixel mixing of raw descriptors first
        self.e0, self.e1, self.e2, self.mid = _block(w[0], w[0]), _block(w[0], w[1]), \
            _block(w[1], w[2]), _block(w[2], w[3])
        self.u2, self.d2 = nn.Conv2d(w[3], w[2], 1), _block(2 * w[2], w[2])
        self.u1, self.d1 = nn.Conv2d(w[2], w[1], 1), _block(2 * w[1], w[1])
        self.u0, self.d0 = nn.Conv2d(w[1], w[0], 1), _block(2 * w[0], w[0])
        self.head = nn.Conv2d(w[0], 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(x, conv, ref):
        return conv(F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False))

    def forward(self, x):
        e0 = self.e0(self.stem(x))
        e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2))
        m = self.mid(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self._up(m, self.u2, e2), e2], 1))
        d1 = self.d1(torch.cat([self._up(d2, self.u1, e1), e1], 1))
        d0 = self.d0(torch.cat([self._up(d1, self.u0, e0), e0], 1))
        out = self.head(d0)
        base = torch.log(self.K0 * torch.expm1(x[:, self.m_idx].float()) + self.B0)
        return (base + out[:, 0]).clamp(-12, 12), out[:, 1].clamp(-6, 3)   # log mu, log alpha


def nb_nll(y, log_mu, log_alpha):
    """Per-pixel negative-binomial NLL (mean parameterization, var = mu + alpha*mu^2)."""
    r = torch.exp(-log_alpha)
    log_r_mu = torch.logaddexp(-log_alpha, log_mu)          # log(r + mu)
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
          + r * (-log_alpha - log_r_mu) + y * (log_mu - log_r_mu))
    return -ll


def regional_log_loss(y, mu, k=8):
    """Smooth-L1 between log totals over k x k blocks (2 degrees at k=8): ties the per-pixel
    model to the regional totals Brasa is scored on."""
    ys = F.avg_pool2d(y[:, None], k, ceil_mode=True) * k * k
    ms = F.avg_pool2d(mu[:, None], k, ceil_mode=True) * k * k
    keep = (ys + ms) > 0.5
    if not keep.any():
        return mu.sum() * 0
    return F.smooth_l1_loss(torch.log1p(ms[keep]), torch.log1p(ys[keep]))


@torch.no_grad()
def predict_month(model, stack, t, device, tile=256, halo=16, override=None, with_alpha=False):
    """Expected VIIRS count map (H, W) for month index t, tiled with overlap.
    with_alpha: also return the NB overdispersion map alpha (var = mu + alpha*mu^2)."""
    H, W = stack.shape
    x = stack.inputs(t, override=override)
    tile = min(tile, -(-max(H, W) // 8) * 8)
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    xp = np.zeros((x.shape[0], Hp + 2 * halo, Wp + 2 * halo), np.float32)
    xp[:, halo:halo + H, halo:halo + W] = x
    out = np.zeros((Hp, Wp), np.float32)
    alpha = np.zeros((Hp, Wp), np.float32)
    for r in range(0, Hp, tile):
        for c in range(0, Wp, tile):
            xb = torch.from_numpy(xp[None, :, r:r + tile + 2 * halo, c:c + tile + 2 * halo]).to(device)
            lm, la = model(xb)
            out[r:r + tile, c:c + tile] = torch.exp(lm[0, halo:-halo, halo:-halo]).cpu().numpy()
            alpha[r:r + tile, c:c + tile] = torch.exp(la[0, halo:-halo, halo:-halo]).cpu().numpy()
    return (out[:H, :W], alpha[:H, :W]) if with_alpha else out[:H, :W]

"""The U-Net, its count likelihood, checkpoints (legacy and v2) and tiled full-map inference.

Every network predicts, per cell, log mu (expected VIIRS S-NPP count) and log alpha (negative-
binomial overdispersion: var = mu + alpha * mu^2), as a residual on the global-ratio baseline:
    log mu = log(K0 * M + B0) + f(x)          M = this month's good MODIS count
f starts at 0 (zero-initialised head), so training starts from Brasa's M0 model.

Checkpoints
  legacy  model/unet.pt from the original ml/train.py: {"state", "width", "n_in", "args", ...};
          loaded with lib_config.LEGACY_SPEC and its state-dict keys renamed (e0 -> enc.0, ...).
  v2      {"format": "brasa-ml/2", "config": ModelConfig dict, "members": [{"state", ...}], ...}:
          one file holds an ensemble (year folds x seeds); predictions average member means.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib_config import LEGACY_SPEC, ModelConfig, parse_years
from lib_stack import Stack

F32 = npt.NDArray[np.float32]
FORMAT_V2 = "brasa-ml/2"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- network --------------------------------------------------------------------------------------
class ChannelNorm(nn.Module):
    """LayerNorm over channels at every pixel: independent of tile size, padding and batch."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], self.weight, self.bias).permute(0, 3, 1, 2)


def _norm(kind: str, c: int) -> nn.Module:
    return ChannelNorm(c) if kind == "layer" else nn.GroupNorm(min(8, c), c)


def _block(i: int, o: int, norm: str = "group") -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), _norm(norm, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), _norm(norm, o), nn.GELU())


class UNet(nn.Module):
    """U-Net with `depth` 2x poolings. depth 3, width 32, norm "group" is the legacy architecture
    (1.8 M parameters)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, n_in = cfg.depth, cfg.spec.n_in
        w = [cfg.width * 2 ** i for i in range(d + 1)]
        self.base_idx: list[int] = cfg.spec.base_index()
        self.stem = nn.Conv2d(n_in, w[0], 1)          # per-pixel mixing of raw descriptors first
        nk = cfg.norm
        self.enc = nn.ModuleList([_block(w[0], w[0], nk)] + [_block(w[i - 1], w[i], nk) for i in range(1, d)])
        self.mid = _block(w[d - 1], w[d], nk)
        self.up = nn.ModuleList([nn.Conv2d(w[i + 1], w[i], 1) for i in range(d)])
        self.dec = nn.ModuleList([_block(2 * w[i], w[i], nk) for i in range(d)])
        self.head = nn.Conv2d(w[0], 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def baseline(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.expm1(x[:, self.base_idx].float()).sum(1)
        return torch.log(self.cfg.k0 * m + self.cfg.b0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, n_in, H, W), H and W multiples of 2^depth -> (log mu, log alpha), each (B, H, W)."""
        skips: list[torch.Tensor] = []
        h = self.stem(x)
        for i, blk in enumerate(self.enc):
            h = blk(h if i == 0 else F.max_pool2d(h, 2))
            skips.append(h)
        h = self.mid(F.max_pool2d(h, 2))
        for i in reversed(range(len(self.dec))):
            ref = skips[i]
            h = self.up[i](F.interpolate(h, size=ref.shape[-2:], mode="bilinear", align_corners=False))
            h = self.dec[i](torch.cat([h, ref], 1))
        out = self.head(h)
        return (self.baseline(x) + out[:, 0]).clamp(-12, 12), out[:, 1].clamp(-6, 3)


def remap_legacy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Original module names (e0, e1, e2, u2, d2, ...) -> UNet ModuleList names (enc.0, up.2, ...)."""
    def key(k: str) -> str:
        k = re.sub(r"^e(\d)\.", r"enc.\1.", k)
        k = re.sub(r"^u(\d)\.", r"up.\1.", k)
        return re.sub(r"^d(\d)\.", r"dec.\1.", k)
    return {key(k): v for k, v in state.items()}


# --- losses ---------------------------------------------------------------------------------------
def nb_nll(y: torch.Tensor, log_mu: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """Per-pixel negative-binomial NLL (mean parameterisation, var = mu + alpha * mu^2)."""
    r = torch.exp(-log_alpha)
    log_r_mu = torch.logaddexp(-log_alpha, log_mu)          # log(r + mu)
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
          + r * (-log_alpha - log_r_mu) + y * (log_mu - log_r_mu))
    return -ll


def regional_log_loss(y: torch.Tensor, mu: torch.Tensor, scales: tuple[int, ...] = (8,)) -> torch.Tensor:
    """Mean over block sizes k of the smooth-L1 between log1p block totals (k = 4/8/20 cells is
    1/2/5 degrees). Ties the per-pixel likelihood to the regional totals the product is scored on."""
    losses: list[torch.Tensor] = []
    for k in scales:
        ys = F.avg_pool2d(y[:, None], k, ceil_mode=True) * k * k
        ms = F.avg_pool2d(mu[:, None], k, ceil_mode=True) * k * k
        keep = (ys + ms) > 0.5
        if keep.any():
            losses.append(F.smooth_l1_loss(torch.log1p(ms[keep]), torch.log1p(ys[keep])))
    return torch.stack(losses).mean() if losses else mu.sum() * 0


# --- inference ------------------------------------------------------------------------------------
@torch.no_grad()
def tiled_forward(model: UNet, x: F32, device: str, tile: int = 256, halo: int | None = None
                  ) -> tuple[F32, F32]:
    """Full-map (log mu, log alpha), each (H, W), for one (n_in, H, W) input, in overlapping tiles."""
    q = 2 ** model.cfg.depth
    halo = model.cfg.halo() if halo is None else halo
    _, H, W = x.shape
    tile = min(tile, -(-max(H, W) // q) * q)
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    xp = np.zeros((x.shape[0], Hp + 2 * halo, Wp + 2 * halo), np.float32)
    xp[:, halo:halo + H, halo:halo + W] = x
    lm = np.zeros((Hp, Wp), np.float32)
    la = np.zeros((Hp, Wp), np.float32)
    for r in range(0, Hp, tile):
        for c in range(0, Wp, tile):
            xb = torch.from_numpy(xp[None, :, r:r + tile + 2 * halo, c:c + tile + 2 * halo]).to(device)
            a, b = model(xb)
            lm[r:r + tile, c:c + tile] = a[0, halo:-halo, halo:-halo].float().cpu().numpy()
            la[r:r + tile, c:c + tile] = b[0, halo:-halo, halo:-halo].float().cpu().numpy()
    return lm[:H, :W], la[:H, :W]


@dataclass
class MonthPrediction:
    """Per-member NB parameters for one month: mu and alpha, each (K, H, W)."""
    mu: F32
    alpha: F32

    @property
    def mean(self) -> F32:
        """Ensemble expected count (mean of member means)."""
        return self.mu.mean(0)

    def sample(self, n: int, rows: npt.NDArray[np.int64], rng: np.random.Generator) -> F32:
        """(n, len(rows)) draws from the member mixture at flat cell indices `rows`."""
        K = self.mu.shape[0]
        mu = self.mu.reshape(K, -1)[:, rows]
        al = self.alpha.reshape(K, -1)[:, rows]
        k = rng.integers(0, K, size=(n, len(rows)))
        m = torch.from_numpy(np.take_along_axis(mu, k, 0).astype(np.float64)).clamp_min(1e-9)
        r = torch.from_numpy(1.0 / np.take_along_axis(al, k, 0).astype(np.float64))
        g = torch.distributions.Gamma(r, r / m).sample()     # NB = Poisson(Gamma) mixture
        return torch.poisson(g).numpy().astype(np.float32)


@dataclass
class Predictor:
    """A loaded checkpoint: one or more UNet members sharing a ModelConfig."""
    name: str
    config: ModelConfig
    members: list[UNet]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def train_years(self) -> list[int]:
        """Every year whose VIIRS data the model saw (training or early stopping)."""
        return sorted(set(self.info.get("train_years", [])) | set(self.info.get("val_years", [])))

    @property
    def is_legacy(self) -> bool:
        return self.info.get("format") != FORMAT_V2

    @classmethod
    def load(cls, path: str | Path, device: str, name: str | None = None) -> Predictor:
        ck: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        name = name or Path(path).stem
        if ck.get("format") == FORMAT_V2:
            cfg = ModelConfig.from_dict(ck["config"])
            states = [m["state"] for m in ck["members"]]
            info = {k: v for k, v in ck.items() if k not in ("members", "config")}
            info["members"] = [{k: v for k, v in m.items() if k != "state"} for m in ck["members"]]
        else:                                               # original ml/train.py checkpoint
            cfg = ModelConfig(spec=LEGACY_SPEC, width=int(ck["width"]), depth=3, norm="group")
            if cfg.spec.n_in != ck["n_in"]:
                raise ValueError(f"legacy checkpoint has n_in={ck['n_in']}, expected {cfg.spec.n_in}")
            states = [remap_legacy_state(ck["state"])]
            args = ck.get("args", {})
            info = {"format": "legacy", "epoch": ck.get("epoch"), "val": ck.get("val"),
                    "train_years": parse_years(args.get("train", "2012-2017")),
                    "val_years": parse_years(args.get("val", "2018"))}
        members = []
        for st in states:
            m = UNet(cfg).to(device)
            m.load_state_dict(st)
            m.eval()
            members.append(m)
        return cls(name=name, config=cfg, members=members, info=info)

    def predict_month(self, stack: Stack, t: int, device: str, aqua_off: bool = False,
                      **input_kw: Any) -> MonthPrediction:
        x = stack.inputs(t, self.config.spec, aqua_off=aqua_off, **input_kw)
        mus, als = [], []
        for m in self.members:
            lm, la = tiled_forward(m, x, device)
            mus.append(np.exp(lm))
            als.append(np.exp(la))
        return MonthPrediction(np.stack(mus), np.stack(als))


def save_checkpoint(path: str | Path, cfg: ModelConfig, members: list[dict[str, Any]],
                    info: dict[str, Any]) -> None:
    """members: [{"state": state_dict, "val_years": [...], "seed": int, "epoch": int, ...}]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT_V2, "config": cfg.to_dict(), "members": members, **info}, path)

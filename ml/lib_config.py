"""Constants, channel catalogue and typed configs shared by every ml/ script.

Nothing in here touches data on disk except `excluded_months()` (reads docs/validation/model.json).

Grid: 0.25 degree cells, cell id = row * 1440 + col, row 0 at 90 S, col 0 at 180 W (same as pipeline/).
Channel names in a stack (lib_stack.py) come from the catalogue below:
  m_*   both MODIS satellites combined (the 15 channels of the original U-Net, "legacy")
  t_*   Terra only, a_* Aqua only (per-satellite blocks of the v2 model)
  aux_* optional external covariates (run_tensor.py --aux)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

REPO: Final[Path] = Path(__file__).resolve().parent.parent
MODEL_JSON: Final[Path] = REPO / "docs" / "validation" / "model.json"

# --- grid ---------------------------------------------------------------------------------------
GRID_COLS: Final[int] = 1440
GRID_ROWS: Final[int] = 720
DEFAULT_ROWS: Final[tuple[int, int]] = (136, 648)    # latitude -56 .. 72
DEFAULT_COLS: Final[tuple[int, int]] = (0, GRID_COLS)
SNAP: Final[int] = 8                                 # bbox stacks are snapped to 2 degrees

# --- detection filters (identical to pipeline/02_aggregate.py) -----------------------------------
MODIS_GOOD: Final[str] = "type = 0 AND TRY_CAST(confidence AS INTEGER) >= 30"
VIIRS_GOOD: Final[str] = "type = 0 AND confidence <> 'l'"

# --- residual baseline of every U-Net: log mu = log(K0 * M + B0) + f(x) ---------------------------
K0: Final[float] = 4.1
B0: Final[float] = 0.05
AQUA_FIRST_MONTH: Final[tuple[int, int]] = (2002, 7)  # fallback when the features table has no a_*

# --- channel catalogue ----------------------------------------------------------------------------
# kind "count": log1p of a non-negative count / sum;  kind "mean": standardized, 0 where undefined
DESCRIPTORS: Final[dict[str, str]] = {
    "n": "count", "n_all": "count", "night": "count", "frp_sum": "count", "frp_max": "count",
    "f10": "count", "f30": "count", "f100": "count", "fbig": "count", "days": "count",
    "maxday": "count", "pix": "mean", "conf": "mean", "bright": "mean", "hour": "mean",
}
LEGACY_CHANNELS: Final[tuple[str, ...]] = (
    "m_n", "m_n_all", "m_terra", "m_night", "m_f10", "m_f30", "m_f100", "m_fbig",
    "m_days", "m_maxday", "m_frp_sum", "m_frp_max", "m_pix", "m_conf", "m_bright",
)
LEGACY_KIND: Final[dict[str, str]] = {c: ("mean" if c in ("m_pix", "m_conf", "m_bright") else "count")
                                      for c in LEGACY_CHANNELS}
SATELLITES: Final[dict[str, str]] = {"t": "Terra", "a": "Aqua"}
SAT_CHANNELS: Final[tuple[str, ...]] = tuple(f"{p}_{d}" for p in SATELLITES for d in DESCRIPTORS)
PRESETS: Final[dict[str, tuple[str, ...]]] = {
    "legacy": LEGACY_CHANNELS,
    "v2": SAT_CHANNELS,
    "all": LEGACY_CHANNELS + SAT_CHANNELS,
}
STATIC_NAMES: Final[tuple[str, ...]] = ("lat", "sin_lon", "cos_lon", "clim_annual")
LOCATION_STATIC: Final[tuple[str, ...]] = ("lat", "sin_lon", "cos_lon")


def channel_kind(name: str) -> str:
    """'count' or 'mean' for any catalogue channel; aux_* channels are standardized like means."""
    if name in LEGACY_KIND:
        return LEGACY_KIND[name]
    if name.startswith("aux_"):
        return "mean"
    prefix, _, desc = name.partition("_")
    if prefix in SATELLITES and desc in DESCRIPTORS:
        return DESCRIPTORS[desc]
    raise KeyError(f"unknown channel {name!r}")


# --- years / months -------------------------------------------------------------------------------
YearMonth = tuple[int, int]


def parse_years(spec: str) -> list[int]:
    """'2012-2017' -> [2012, ..., 2017]; '2018' -> [2018]; '2012-2014,2019' -> union; 'none' -> []."""
    if spec.strip().lower() in ("", "none"):
        return []
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def parse_month(spec: str) -> YearMonth:
    """'2002-07' -> (2002, 7)."""
    y, m = spec.split("-")
    return int(y), int(m)


def month_str(ym: YearMonth) -> str:
    return f"{ym[0]}-{ym[1]:02d}"


def excluded_months() -> list[str]:
    """Known VIIRS outage months ('YYYY-MM'), excluded from training and scoring everywhere."""
    if not MODEL_JSON.exists():
        return []
    return list(json.loads(MODEL_JSON.read_text())["excluded_training_months"])


# --- typed configs stored inside checkpoints ------------------------------------------------------
@dataclass(frozen=True)
class InputSpec:
    """Which stack channels a model reads and how they are laid out along the channel axis.

    Layout (lib_stack.Stack.inputs): for each offset in `offsets`, every name in `channels`;
    then `static` maps, then the calendar-month MODIS climatology, then sin/cos(month), then the
    Aqua-availability flag. The legacy U-Net is offsets (-1, 0, 1), LEGACY_CHANNELS, all statics,
    climatology and season, no flag: 3 * 15 + 4 + 1 + 2 = 52 inputs.
    """
    channels: tuple[str, ...]
    offsets: tuple[int, ...] = (-1, 0, 1)
    static: tuple[str, ...] = STATIC_NAMES
    clim_month: bool = True
    season: bool = True
    aqua_flag: bool = False
    base_channels: tuple[str, ...] = ("m_n",)   # current-month counts summed into the baseline M

    @property
    def n_in(self) -> int:
        return (len(self.channels) * len(self.offsets) + len(self.static)
                + int(self.clim_month) + 2 * int(self.season) + int(self.aqua_flag))

    def base_index(self) -> list[int]:
        """Input-channel indices of the current-month (offset 0) baseline counts."""
        o = self.offsets.index(0) * len(self.channels)
        return [o + self.channels.index(c) for c in self.base_channels]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> InputSpec:
        return InputSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


LEGACY_SPEC: Final[InputSpec] = InputSpec(channels=LEGACY_CHANNELS)


NormKind = Literal["layer", "group"]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of one U-Net member. depth = number of 2x poolings.
    norm "layer": per-pixel LayerNorm over channels, so a cell's prediction does not depend on the
    tile it is computed in (patch training == full-map inference). "group": GroupNorm over the whole
    tile, as in the legacy model, which makes outputs depend on tile size and zero padding."""
    spec: InputSpec
    width: int = 32
    depth: int = 3
    norm: NormKind = "layer"
    k0: float = K0
    b0: float = B0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ModelConfig:
        d = dict(d)
        d["spec"] = InputSpec.from_dict(d["spec"])
        d.setdefault("norm", "group")
        return ModelConfig(**d)

    def receptive_radius(self) -> int:
        """Approximate receptive radius in cells: two 3x3 convs per block at each resolution,
        encoder + decoder, plus pooling. depth 3 -> 51 cells (~13 degrees)."""
        return 5 * (2 ** self.depth - 1) + 2 ** (self.depth + 1)

    def halo(self) -> int:
        """Tile overlap for full-map inference: >= receptive radius, multiple of 8 and of 2^depth.
        (At 16 cells the legacy model's tile seams shifted global totals by 1.8%.)"""
        q = max(8, 2 ** self.depth)
        return max(48, int(math.ceil(self.receptive_radius() / q)) * q)


SamplingMode = Literal["input", "target"]


@dataclass
class TrainConfig:
    """Everything run_train.py needs besides the model architecture (stored in the checkpoint)."""
    train_years: list[int]
    val_years: list[int]
    folds: int = 1
    seeds: int = 1
    patch: int = 128
    batch: int = 16
    epochs: int = 40
    steps: int = 500
    lr: float = 2e-3
    weight_decay: float = 1e-4
    reg_w: float = 0.3
    reg_scales: tuple[int, ...] = (4, 8, 20)
    sampling: SamplingMode = "input"
    aqua_dropout: float = 0.15
    val_patches: int = 512
    val_months: int = 12
    val_every: int = 1
    patience: int = 8
    select: Literal["metric", "nll"] = "metric"
    workers: int = 4
    seed: int = 0

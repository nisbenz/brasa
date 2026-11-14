"""Read access to one tensor stack written by run_tensor.py, and model-input assembly.

A stack directory holds memory-mapped arrays (T = months, C = channels, H x W = grid window):
  X.npy   (T, C, H, W) float16   MODIS channels, names in meta["channels"] (lib_config catalogue)
  Y.npy   (T, H, W)    float32   VIIRS S-NPP good detections (target; NaN before 2012)
  N20.npy (T, H, W)    float32   VIIRS NOAA-20 good detections (ceiling; NaN before 2018)
  S.npy   (s, H, W)    float32   static maps, names in meta["static"]
  CL.npy  (12, H, W)   float32   log1p MODIS climatology per calendar month
  meta.json                      grid window, months, channels, normalization, Aqua months, ...
Models never index channels by position: they carry an InputSpec of channel *names*, so one stack
built with --preset all serves both the legacy U-Net and the v2 model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lib_config import GRID_COLS, InputSpec, YearMonth, month_str

F32 = npt.NDArray[np.float32]
Window = tuple[int, int, int, int]   # r0, r1, c0, c1 relative to the stack window


class Stack:
    """Lazy memmap view of a stack directory. Safe to pickle into DataLoader workers."""

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self.meta: dict[str, Any] = json.loads((self.path / "meta.json").read_text())
        self.months: list[YearMonth] = [tuple(map(int, s.split("-"))) for s in self.meta["months"]]  # type: ignore[misc]
        self.ordinal: npt.NDArray[np.int64] = np.array([y * 12 + m - 1 for y, m in self.months])
        self.channels: tuple[str, ...] = tuple(self.meta["channels"])
        self.static_names: tuple[str, ...] = tuple(self.meta["static"])
        self._ch: dict[str, int] = {c: i for i, c in enumerate(self.channels)}
        self._st: dict[str, int] = {c: i for i, c in enumerate(self.static_names)}
        self._t: dict[YearMonth, int] = {ym: i for i, ym in enumerate(self.months)}
        self.aqua_months: set[str] = set(self.meta.get("aqua_months", []))
        self.excluded: set[str] = set(self.meta.get("excluded_months", []))
        self._arr: dict[str, np.ndarray] | None = None

    def __getstate__(self) -> dict[str, Any]:
        s = self.__dict__.copy()
        s["_arr"] = None
        return s

    # --- raw arrays -------------------------------------------------------------------------------
    @property
    def arr(self) -> dict[str, np.ndarray]:
        if self._arr is None:
            self._arr = {k: np.load(self.path / f"{k}.npy", mmap_mode="r")
                         for k in ("X", "Y", "N20", "S", "CL")}
        return self._arr

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.arr["Y"].shape[1:])  # type: ignore[return-value]

    @property
    def origin(self) -> tuple[int, int]:
        """(row, col) of the stack's top-left cell in the global 720 x 1440 grid."""
        return int(self.meta["r0"]), int(self.meta["c0"])

    def cells(self) -> npt.NDArray[np.int64]:
        """(H, W) global cell ids of the stack window."""
        r0, c0 = self.origin
        H, W = self.shape
        return (np.arange(r0, r0 + H)[:, None] * GRID_COLS + np.arange(c0, c0 + W)[None]).astype(np.int64)

    # --- months -----------------------------------------------------------------------------------
    def index(self, ym: YearMonth) -> int | None:
        return self._t.get(ym)

    def neighbour(self, t: int, dt: int) -> int | None:
        """Month index of t + dt calendar months, if that month is in the stack."""
        if dt == 0:
            return t
        u = t + dt
        if 0 <= u < len(self.months) and self.ordinal[u] == self.ordinal[t] + dt:
            return u
        y, m = divmod(int(self.ordinal[t]) + dt, 12)
        return self._t.get((y, m + 1))

    def has_aqua(self, t: int) -> bool:
        return month_str(self.months[t]) in self.aqua_months

    def usable(self, years: set[int] | list[int]) -> list[int]:
        """Month indices in `years` with S-NPP data, excluding known VIIRS outage months."""
        ys = set(years)
        return [i for i, (y, m) in enumerate(self.months)
                if y in ys and y >= 2012 and f"{y}-{m:02d}" not in self.excluded]

    # --- channels ---------------------------------------------------------------------------------
    def check(self, spec: InputSpec) -> None:
        """Raise with a readable message if the stack lacks a channel or static map the spec needs."""
        miss = [c for c in spec.channels if c not in self._ch] + \
               [c for c in spec.static if c not in self._st]
        if miss:
            raise SystemExit(f"stack {self.path} lacks {miss}; rebuild it with run_tensor.py --preset all")

    def channel(self, t: int, name: str, win: Window | None = None) -> F32:
        r0, r1, c0, c1 = win or (0, self.shape[0], 0, self.shape[1])
        return np.asarray(self.arr["X"][t, self._ch[name], r0:r1, c0:c1], np.float32)

    def modis_count(self, t: int, win: Window | None = None) -> F32:
        """Good MODIS detections (both satellites) per cell for month t."""
        if "m_n" in self._ch:
            return np.expm1(self.channel(t, "m_n", win))
        return np.expm1(self.channel(t, "t_n", win)) + np.expm1(self.channel(t, "a_n", win))

    def inputs(self, t: int, spec: InputSpec, win: Window | None = None, aqua_off: bool = False,
               swap: dict[str, int] | None = None, static_shift: tuple[int, int] | None = None) -> F32:
        """(spec.n_in, h, w) float32 model inputs for month index t (layout: lib_config.InputSpec).

        aqua_off      zero every a_* channel and the Aqua flag (Terra-only simulation / inference)
        swap          {channel: month index}: read that channel from another month (permutation tests)
        static_shift  roll static and climatology maps by (rows, cols) (permutation tests)
        """
        a = self.arr
        H, W = self.shape
        r0, r1, c0, c1 = win or (0, H, 0, W)
        h, w = r1 - r0, c1 - c0
        swap = swap or {}
        if aqua_off and any(c.startswith("m_") for c in spec.channels):
            raise ValueError("aqua_off needs per-satellite channels; m_* already include Aqua")
        idx = [self._ch[c] for c in spec.channels]
        parts: list[F32] = []
        for dt in spec.offsets:
            u = self.neighbour(t, dt)
            if u is None:
                parts.append(np.zeros((len(idx), h, w), np.float32))
                continue
            x = np.asarray(a["X"][u, idx, r0:r1, c0:c1], np.float32)
            for k, c in enumerate(spec.channels):
                if c in swap:
                    v = self.neighbour(swap[c], dt)
                    x[k] = 0.0 if v is None else a["X"][v, self._ch[c], r0:r1, c0:c1]
                if aqua_off and c.startswith("a_"):
                    x[k] = 0.0
            parts.append(x)
        m = self.months[t][1]
        st = [np.asarray(a["S"][self._st[c]], np.float32) for c in spec.static]
        cl = [np.asarray(a["CL"][m - 1], np.float32)] if spec.clim_month else []
        maps = st + cl
        if static_shift:
            maps = [np.roll(v, static_shift, axis=(0, 1)) for v in maps]
        parts += [v[None, r0:r1, c0:c1] for v in maps]
        if spec.season:
            ang = 2 * np.pi * (m - 0.5) / 12
            parts.append(np.broadcast_to(np.array([np.sin(ang), np.cos(ang)], np.float32)[:, None, None],
                                         (2, h, w)))
        if spec.aqua_flag:
            on = float(self.has_aqua(t) and not aqua_off)
            parts.append(np.full((1, h, w), on, np.float32))
        return np.ascontiguousarray(np.concatenate(parts, 0), dtype=np.float32)

    # --- targets ----------------------------------------------------------------------------------
    def target(self, t: int, win: Window | None = None) -> F32:
        """VIIRS S-NPP counts (NaN where S-NPP does not exist)."""
        r0, r1, c0, c1 = win or (0, self.shape[0], 0, self.shape[1])
        return np.array(self.arr["Y"][t, r0:r1, c0:c1], np.float32)

    def ceiling(self, t: int, win: Window | None = None) -> F32:
        """VIIRS NOAA-20 counts (NaN before 2018): a second instrument, used as the noise ceiling."""
        r0, r1, c0, c1 = win or (0, self.shape[0], 0, self.shape[1])
        return np.array(self.arr["N20"][t, r0:r1, c0:c1], np.float32)

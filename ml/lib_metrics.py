"""Scoring shared by run_train.py (validation) and run_evaluate.py (model comparison).

Everything works on dense arrays: a model's prediction for a set of months is aggregated to
region totals P[t, r] (months x regions) once, and every metric is computed from P and the
observed totals O. `score()` reproduces pipeline/03_validate.py (and the old ml/evaluate.py)
exactly; test_lib.py checks that against the original pandas implementation.

Scales: "1deg" and "5deg" use pipeline/harmonize.parents(); "country" uses cell_country.parquet
(cells without a country form one "?" region, as before).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from lib_config import REPO

sys.path.insert(0, str(REPO / "pipeline"))
from harmonize import fit as brasa_fit, parents, ratio  # noqa: E402

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
LOG15 = float(np.log(1.5))
LOG125 = float(np.log(1.25))


# --- regions --------------------------------------------------------------------------------------
@dataclass
class Regions:
    """Compact region ids (0..n-1) for every cell of a stack window, per scale."""
    ids: dict[str, I64]          # scale -> (H*W,) region id
    n: dict[str, int]            # scale -> number of regions
    names: dict[str, list[str]]  # scale -> region label (country names; numeric ids otherwise)

    @classmethod
    def build(cls, cells: I64, cell_country: Path | None = None,
              scales: tuple[str, ...] = ("1deg", "5deg", "country")) -> Regions:
        flat = cells.reshape(-1)
        raw: dict[str, npt.NDArray[Any]] = {}
        p1, p5 = parents(flat)
        if "1deg" in scales:
            raw["1deg"] = p1
        if "5deg" in scales:
            raw["5deg"] = p5
        if "country" in scales:
            if cell_country is None:
                raise ValueError("country scale needs cell_country.parquet")
            cc = pd.read_parquet(cell_country).set_index("cell").country
            raw["country"] = cc.reindex(flat).fillna("?").to_numpy()
        ids, n, names = {}, {}, {}
        for sc, v in raw.items():
            u, inv = np.unique(v, return_inverse=True)
            ids[sc], n[sc], names[sc] = inv.astype(np.int64), len(u), [str(x) for x in u]
        return cls(ids, n, names)

    def sums(self, m: npt.NDArray[Any]) -> dict[str, F64]:
        """Region totals of one (H, W) map, NaN treated as 0."""
        w = np.nan_to_num(np.asarray(m, np.float64).reshape(-1))
        return {sc: np.bincount(i, w, self.n[sc]) for sc, i in self.ids.items()}


class RegionSeries:
    """Accumulates region totals month by month for one source (model, Brasa, observation)."""

    def __init__(self, regions: Regions) -> None:
        self.regions = regions
        self.rows: dict[str, list[F64]] = {sc: [] for sc in regions.ids}

    def add(self, m: npt.NDArray[Any]) -> None:
        for sc, v in self.regions.sums(m).items():
            self.rows[sc].append(v)

    def array(self, scale: str) -> F64:
        return np.stack(self.rows[scale]) if self.rows[scale] else np.zeros((0, self.regions.n[scale]))


# --- the pipeline/03_validate.py metrics ----------------------------------------------------------
def score(P: F64, O: F64, cal_month: npt.NDArray[np.integer[Any]]) -> dict[str, float]:
    """Metrics of predicted vs observed region totals, P and O (months x regions), cal_month (months,).

    bias_pct            100 * (sum P / sum O - 1)
    med_abs_log_err     median |log((p+1)/(o+1))| over region-months with p + o >= 20
    within_25pct        share of those within 25%
    r_log               correlation of log1p(p), log1p(o) over region-months with p + o > 0
    anomaly_*           anomaly = log((x+1)/(median of that region and calendar month + 1)), over
                        regions with >= 200 observed detections; sign agreement where |o anomaly| >
                        log 1.5; anomaly_slope = OLS slope of predicted on observed anomaly (1 = right
                        amplitude, < 1 = damped year-to-year variability)
    """
    nan = float("nan")
    nz = (P + O) > 0
    big = (P + O) >= 20
    le = np.log((P[big] + 1) / (O[big] + 1))
    out: dict[str, float] = {
        "bias_pct": round(100 * (P[nz].sum() / O[nz].sum() - 1), 2) if O[nz].sum() > 0 else nan,
        "med_abs_log_err": round(float(np.median(np.abs(le))), 3) if le.size else nan,
        "within_25pct": round(float((np.abs(le) < LOG125).mean()), 3) if le.size else nan,
        "r_log": round(float(np.corrcoef(np.log1p(P[nz]), np.log1p(O[nz]))[0, 1]), 4) if nz.sum() > 2 else nan,
    }
    keep = (O * nz).sum(0) >= 200
    p, o = P[:, keep], O[:, keep]
    an: dict[str, F64] = {}
    if p.size:
        for name, x in (("p", p), ("o", o)):
            med = np.zeros_like(x)
            for mo in np.unique(cal_month):
                sel = cal_month == mo
                med[sel] = np.median(x[sel], axis=0)
            an[name] = np.log((x + 1) / (med + 1))
    if an and an["o"].size > 2:
        pa, oa = an["p"].reshape(-1), an["o"].reshape(-1)
        strong = np.abs(oa) > LOG15
        with np.errstate(invalid="ignore", divide="ignore"):
            out["anomaly_r"] = round(float(np.corrcoef(pa, oa)[0, 1]), 4)
            out["anomaly_sign_agree"] = round(float((np.sign(pa) == np.sign(oa))[strong].mean()), 3) \
                if strong.any() else nan
            out["anomaly_slope"] = round(float(np.polyfit(oa, pa, 1)[0]), 3) if oa.std() > 0 else nan
    else:
        out.update(anomaly_r=nan, anomaly_sign_agree=nan, anomaly_slope=nan)
    return out


def bootstrap_diff(PA: F64, PB: F64, O: F64, n: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap over months of median |log err|(A) - median |log err|(B) on region-months with
    >= 20 observed detections (same rows for both). Returns (point, 2.5%, 97.5%); negative = A better."""
    rows = O >= 20
    ea = np.abs(np.log((PA + 1) / (O + 1)))
    eb = np.abs(np.log((PB + 1) / (O + 1)))
    per_a = [ea[t][rows[t]] for t in range(len(O))]
    per_b = [eb[t][rows[t]] for t in range(len(O))]
    point = float(np.median(np.concatenate(per_a)) - np.median(np.concatenate(per_b)))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        ix = rng.integers(0, len(O), len(O))
        d.append(np.median(np.concatenate([per_a[i] for i in ix]))
                 - np.median(np.concatenate([per_b[i] for i in ix])))
    return round(point, 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def gap_closed(model_err: float, brasa_err: float, ceiling_err: float) -> float:
    """Share of the Brasa -> ceiling error gap the model closes (0 = Brasa, 1 = ceiling)."""
    gap = brasa_err - ceiling_err
    return round((brasa_err - model_err) / gap, 3) if gap > 0 else float("nan")


# --- calibration ----------------------------------------------------------------------------------
def randomized_pit(samples: F32, y: F32, rng: np.random.Generator) -> F64:
    """Randomized PIT of integer observations y (cells,) under predictive draws samples (n, cells):
    u = P(Y < y) + v * P(Y = y), v ~ U(0, 1). Uniform u = calibrated; U-shape = overconfident."""
    below = (samples < y[None]).mean(0)
    at = (samples == y[None]).mean(0)
    return below + rng.random(len(y)) * at


def pit_summary(u: F64, bins: int = 10) -> dict[str, Any]:
    h = np.histogram(u, bins=bins, range=(0, 1))[0]
    return {"n": int(len(u)), "coverage_80": round(float(((u > 0.1) & (u < 0.9)).mean()), 3),
            "pit_mean": round(float(u.mean()), 3),
            "pit_hist": [round(float(x), 3) for x in h / max(len(u), 1) * bins]}


# --- Brasa M3 (pipeline/harmonize.py) as maps -----------------------------------------------------
@dataclass
class BrasaMaps:
    """Brasa M3 refit, precomputed for a stack window: ratio K[month] and small-fire floor B[month]."""
    K: F32   # (12, H, W) VIIRS / MODIS ratio
    B: F32   # (12, H, W) expected VIIRS count where MODIS saw nothing (active cells only)

    @classmethod
    def fit(cls, train: pd.DataFrame, cells: I64, a: float = 20.0) -> BrasaMaps:
        """train: cell, year, month, M, V rows with M > 0 or V > 0 (as pipeline/03_validate.py)."""
        model = brasa_fit(train, a=a, seasonal=True, floor=True)
        H, W = cells.shape
        flat = cells.reshape(-1)
        K = np.stack([ratio(model, flat, np.full(flat.shape, m)).reshape(H, W) for m in range(1, 13)])
        B = np.zeros((12, H * W), np.float32)
        act = model["active"]
        pos = pd.Series(np.arange(flat.size), index=flat)
        hit = act.index.intersection(pos.index)
        b = model["b"]
        for m in range(1, 13):
            val = b.reindex(pd.MultiIndex.from_arrays([act.loc[hit].to_numpy(), np.full(len(hit), m)]))
            B[m - 1, pos.loc[hit].to_numpy()] = np.nan_to_num(val.to_numpy())
        return cls(K.astype(np.float32), B.reshape(12, H, W))

    def predict(self, M: F32, month: int) -> F32:
        return np.where(M > 0, self.K[month - 1] * M, self.B[month - 1]).astype(np.float32)


def load_brasa_train(features: Path, years: list[int], excluded: set[str],
                     window: tuple[int, int, int, int], mcol: str = "m_n") -> pd.DataFrame:
    """Brasa training rows from features.parquet, restricted to the stack window and years.
    mcol = "t_n" fits the Terra-only variant (the reference for simulated Terra-only months)."""
    import duckdb
    r0, r1, c0, c1 = window
    df = duckdb.sql(f"""
        SELECT cell, year, month, coalesce({mcol}, 0)::DOUBLE AS M, coalesce(v_n, 0)::DOUBLE AS V
        FROM '{features}'
        WHERE year IN ({",".join(map(str, years))}) AND (coalesce({mcol}, 0) > 0 OR coalesce(v_n, 0) > 0)
          AND cell // 1440 >= {r0} AND cell // 1440 < {r1} AND cell % 1440 >= {c0} AND cell % 1440 < {c1}
    """).df()
    bad = np.array([f"{y}-{m:02d}" in excluded for y, m in zip(df.year, df.month)], bool)
    return df[~bad].reset_index(drop=True)


# --- fast validation metric (run_train.py) -------------------------------------------------------
def map_score(pred_fn: Callable[[int], F32], stack_target: Callable[[int], F32], months: list[int],
              cal_months: list[int], regions: Regions, scale: str = "5deg") -> dict[str, float]:
    """score() at one scale for full-map predictions of `months` (month indices)."""
    P, O = RegionSeries(regions), RegionSeries(regions)
    for t in months:
        P.add(pred_fn(t))
        O.add(stack_target(t))
    return score(P.array(scale), O.array(scale), np.array(cal_months))


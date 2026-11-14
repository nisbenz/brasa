"""Unit tests for the shared ml/ libraries (stdlib unittest; no pytest needed).

Usage: python -I ml/test_lib.py [-v]
       BRASA_DERIVED=~/firms/derived python -I ml/test_lib.py     # also run the slow data checks
Fast tests build a tiny synthetic stack in a temp dir. Data checks (skipped unless BRASA_DERIVED is
set) load model/unet.pt and confirm the legacy model still loads and predicts sane totals.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from lib_config import (LEGACY_SPEC, PRESETS, REPO, STATIC_NAMES, InputSpec, ModelConfig,  # noqa: E402
                        channel_kind, parse_years)
from lib_metrics import BrasaMaps, bootstrap_diff, gap_closed, randomized_pit, score  # noqa: E402
from lib_model import (MonthPrediction, Predictor, UNet, nb_nll, regional_log_loss,  # noqa: E402
                       remap_legacy_state, save_checkpoint, tiled_forward)
from lib_stack import Stack  # noqa: E402

V2_SPEC = InputSpec(channels=PRESETS["v2"], aqua_flag=True, base_channels=("t_n", "a_n"))


def make_stack(root: Path, H: int = 32, W: int = 48, months: list[str] | None = None) -> Stack:
    """Synthetic stack with every catalogue channel; Aqua present from 2002-07."""
    months = months or ["2002-05", "2002-06", "2002-07", "2002-08", "2002-10"]   # 2002-09 missing
    ch = list(PRESETS["all"])
    rng = np.random.default_rng(0)
    T = len(months)
    X = rng.random((T, len(ch), H, W)).astype(np.float16)
    np.save(root / "X.npy", X)
    np.save(root / "Y.npy", rng.poisson(3, (T, H, W)).astype(np.float32))
    np.save(root / "N20.npy", rng.poisson(3, (T, H, W)).astype(np.float32))
    np.save(root / "S.npy", rng.random((len(STATIC_NAMES), H, W)).astype(np.float32))
    np.save(root / "CL.npy", rng.random((12, H, W)).astype(np.float32))
    json.dump({"r0": 300, "r1": 300 + H, "c0": 600, "c1": 600 + W, "months": months, "channels": ch,
               "static": list(STATIC_NAMES), "aqua_months": [m for m in months if m >= "2002-07"],
               "excluded_months": []}, open(root / "meta.json", "w"))
    return Stack(root)


def legacy_score(pred: pd.Series, obs: pd.Series, months: list[tuple[int, int]]) -> dict[str, float]:
    """The original pandas score() of ml/evaluate.py, kept verbatim as the reference."""
    j = pd.concat([pred.rename("p"), obs.rename("o")], axis=1).fillna(0)
    j = j[[ym in months for ym in zip(j.index.get_level_values(1), j.index.get_level_values(2))]]
    j = j[(j.p + j.o) > 0]
    big = j[(j.p + j.o) >= 20]
    le = np.log((big.p + 1) / (big.o + 1))
    k = j.reset_index()
    k = k[k.groupby("region").o.transform("sum") >= 200]
    regions = k.region.unique()
    grid = pd.MultiIndex.from_tuples([(r, y, m) for r in regions for y, m in months],
                                     names=["region", "year", "month"])
    full = k.set_index(["region", "year", "month"]).reindex(grid).fillna(0).reset_index()
    for col in ("p", "o"):
        med = full.groupby(["region", "month"])[col].transform("median")
        full[col + "_an"] = np.log((full[col] + 1) / (med + 1))
    strong = np.abs(full.o_an) > np.log(1.5)
    return dict(
        bias_pct=round(100 * (j.p.sum() / j.o.sum() - 1), 2),
        med_abs_log_err=round(float(np.median(np.abs(le))), 3),
        within_25pct=round(float((np.abs(le) < np.log(1.25)).mean()), 3),
        r_log=round(float(np.corrcoef(np.log1p(j.p), np.log1p(j.o))[0, 1]), 4),
        anomaly_r=round(float(np.corrcoef(full.p_an, full.o_an)[0, 1]), 4),
        anomaly_sign_agree=round(float((np.sign(full.p_an) == np.sign(full.o_an))[strong].mean()), 3),
    )


class TestConfig(unittest.TestCase):
    def test_years(self) -> None:
        self.assertEqual(parse_years("2012-2014,2019"), [2012, 2013, 2014, 2019])
        self.assertEqual(parse_years("none"), [])

    def test_legacy_layout(self) -> None:
        self.assertEqual(LEGACY_SPEC.n_in, 52)          # 3 * 15 + 4 static + 1 climatology + 2 season
        self.assertEqual(LEGACY_SPEC.base_index(), [15])
        self.assertEqual(V2_SPEC.n_in, 3 * 30 + 4 + 1 + 2 + 1)
        self.assertEqual(V2_SPEC.base_index(), [30, 30 + 15])

    def test_kinds(self) -> None:
        self.assertEqual(channel_kind("m_pix"), "mean")
        self.assertEqual(channel_kind("a_hour"), "mean")
        self.assertEqual(channel_kind("t_frp_max"), "count")
        self.assertEqual(channel_kind("aux_burned"), "mean")

    def test_halo(self) -> None:
        cfg = ModelConfig(spec=LEGACY_SPEC)
        self.assertGreaterEqual(cfg.halo(), cfg.receptive_radius())
        self.assertEqual(cfg.halo() % 8, 0)


class TestStack(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.s = make_stack(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_layout_and_neighbours(self) -> None:
        x = self.s.inputs(2, V2_SPEC)
        self.assertEqual(x.shape, (V2_SPEC.n_in, 32, 48))
        np.testing.assert_allclose(x[30], self.s.channel(2, "t_n"))             # offset 0 block
        np.testing.assert_allclose(x[0], self.s.channel(1, "t_n"))              # offset -1 block
        x3 = self.s.inputs(3, V2_SPEC)                                           # 2002-08: next is missing
        self.assertTrue((x3[60:90] == 0).all())
        self.assertEqual(self.s.neighbour(4, -1), None)                          # 2002-10 -> 2002-09 absent

    def test_aqua_flag_and_dropout(self) -> None:
        self.assertEqual(self.s.inputs(0, V2_SPEC)[-1].max(), 0.0)              # 2002-05: Terra only
        x = self.s.inputs(2, V2_SPEC)
        self.assertEqual(x[-1].min(), 1.0)
        off = self.s.inputs(2, V2_SPEC, aqua_off=True)
        a_idx = [k for k, c in enumerate(V2_SPEC.channels) if c.startswith("a_")]
        self.assertTrue((off[[30 + i for i in a_idx]] == 0).all())
        self.assertEqual(off[-1].max(), 0.0)
        t_idx = [k for k, c in enumerate(V2_SPEC.channels) if c.startswith("t_")]
        np.testing.assert_array_equal(off[[30 + i for i in t_idx]], x[[30 + i for i in t_idx]])
        with self.assertRaises(ValueError):
            self.s.inputs(2, LEGACY_SPEC, aqua_off=True)

    def test_swap_and_window(self) -> None:
        x = self.s.inputs(2, V2_SPEC, swap={"t_n": 3})
        np.testing.assert_allclose(x[30], self.s.channel(3, "t_n"))
        w = self.s.inputs(2, V2_SPEC, win=(4, 12, 8, 24))
        np.testing.assert_allclose(w, self.s.inputs(2, V2_SPEC)[:, 4:12, 8:24])

    def test_modis_count(self) -> None:
        np.testing.assert_allclose(self.s.modis_count(1), np.expm1(self.s.channel(1, "m_n")))


class TestModel(unittest.TestCase):
    def test_nb_nll_matches_torch(self) -> None:
        y = torch.tensor([0.0, 3.0, 17.0])
        lm, la = torch.log(torch.tensor([2.0, 4.0, 10.0])), torch.log(torch.tensor([0.5, 0.1, 1.0]))
        r = torch.exp(-la)
        ref = -torch.distributions.NegativeBinomial(r, probs=torch.exp(lm) / (r + torch.exp(lm))).log_prob(y)
        torch.testing.assert_close(nb_nll(y, lm, la), ref, rtol=1e-4, atol=1e-4)

    def test_regional_loss_zero_when_exact(self) -> None:
        y = torch.rand(2, 40, 40) * 5
        self.assertAlmostEqual(regional_log_loss(y, y, (4, 8, 20)).item(), 0.0, places=6)

    def test_unet_starts_at_baseline(self) -> None:
        m = UNet(ModelConfig(spec=V2_SPEC, width=8, depth=2)).eval()
        x = torch.rand(1, V2_SPEC.n_in, 32, 32)
        lm, la = m(x)
        torch.testing.assert_close(lm, m.baseline(x).clamp(-12, 12))
        self.assertTrue((la == 0).all())

    def test_legacy_remap_strict(self) -> None:
        names = ["e0.0.weight", "u2.weight", "d1.3.bias", "mid.0.weight", "stem.bias", "head.weight"]
        self.assertEqual(list(remap_legacy_state({k: torch.zeros(1) for k in names})),
                         ["enc.0.0.weight", "up.2.weight", "dec.1.3.bias", "mid.0.weight", "stem.bias", "head.weight"])
        legacy = REPO / "model" / "unet.pt"
        if legacy.exists():
            p = Predictor.load(legacy, "cpu", "old")              # load_state_dict is strict
            self.assertTrue(p.is_legacy)
            self.assertEqual(p.train_years, list(range(2012, 2019)))

    def test_tiled_single_tile_equals_forward(self) -> None:
        m = UNet(ModelConfig(spec=V2_SPEC, width=8, depth=2))
        torch.nn.init.normal_(m.head.weight, std=0.1)
        m.eval()
        x = np.random.default_rng(1).random((V2_SPEC.n_in, 40, 40)).astype(np.float32)
        lm, _ = tiled_forward(m, x, "cpu", tile=64, halo=12)      # tile shrinks to the 40 x 40 map
        xp = np.zeros((V2_SPEC.n_in, 40 + 24, 40 + 24), np.float32)
        xp[:, 12:52, 12:52] = x
        ref = m(torch.from_numpy(xp)[None])[0][0, 12:52, 12:52].detach().numpy()
        np.testing.assert_allclose(lm, ref, rtol=1e-5, atol=1e-5)

    def test_layer_norm_is_tile_independent(self) -> None:
        m = UNet(ModelConfig(spec=V2_SPEC, width=8, depth=2, norm="layer"))
        torch.nn.init.normal_(m.head.weight, std=0.1)
        m.eval()
        x = np.random.default_rng(2).random((V2_SPEC.n_in, 96, 96)).astype(np.float32)
        small, _ = tiled_forward(m, x, "cpu", tile=32)                 # 9 tiles, halo >= receptive radius
        big, _ = tiled_forward(m, x, "cpu", tile=96)                   # one tile
        np.testing.assert_allclose(small, big, rtol=1e-4, atol=1e-4)

    def test_checkpoint_roundtrip_ensemble(self) -> None:
        cfg = ModelConfig(spec=V2_SPEC, width=8, depth=2)
        nets = [UNet(cfg) for _ in range(2)]
        for n in nets:
            torch.nn.init.normal_(n.head.weight, std=0.1)
        with tempfile.TemporaryDirectory() as d:
            s = make_stack(Path(d))
            path = Path(d) / "m.pt"
            save_checkpoint(path, cfg, [{"state": n.state_dict(), "seed": i} for i, n in enumerate(nets)],
                            {"train_years": [2012], "val_years": [2013]})
            p = Predictor.load(path, "cpu")
            self.assertFalse(p.is_legacy)
            pr = p.predict_month(s, 2, "cpu")
            self.assertEqual(pr.mu.shape, (2, 32, 48))
            np.testing.assert_allclose(pr.mean, pr.mu.mean(0))
            self.assertEqual(p.train_years, [2012, 2013])

    def test_mixture_sampling_mean(self) -> None:
        mu = np.full((2, 1, 50), 5.0, np.float32)
        mu[1] = 15.0
        pr = MonthPrediction(mu, np.full_like(mu, 0.3))
        d = pr.sample(4000, np.arange(50), np.random.default_rng(0))
        self.assertAlmostEqual(float(d.mean()), 10.0, delta=0.3)


class TestMetrics(unittest.TestCase):
    def test_score_matches_legacy_pandas(self) -> None:
        rng = np.random.default_rng(3)
        months = [(y, m) for y in (2019, 2020, 2021) for m in range(1, 13)]
        R = 40
        O = rng.negative_binomial(1, 0.02, (len(months), R)).astype(float)
        O[:, :5] = 0                                                   # some empty regions
        P = np.round(O * np.exp(rng.normal(0, 0.3, O.shape)))
        idx = pd.MultiIndex.from_tuples([(r, y, m) for (y, m) in months for r in range(R)],
                                        names=["region", "year", "month"])
        ps, os_ = pd.Series(P.reshape(-1), idx), pd.Series(O.reshape(-1), idx)
        ref = legacy_score(ps[ps > 0], os_[os_ > 0], months)
        got = score(P, O, np.array([m for _, m in months]))
        for k, v in ref.items():
            self.assertAlmostEqual(got[k], v, places=6, msg=k)

    def test_bootstrap_and_gap(self) -> None:
        rng = np.random.default_rng(0)
        O = rng.poisson(100, (24, 30)).astype(float)
        good, bad = O * 1.05, O * 1.6
        d, lo, hi = bootstrap_diff(good, bad, O, 200)
        self.assertLess(hi, 0)
        self.assertEqual(gap_closed(0.2, 0.3, 0.1), 0.5)

    def test_pit_calibrated(self) -> None:
        rng = np.random.default_rng(0)
        pr = MonthPrediction(np.full((1, 1, 20000), 6.0, np.float32), np.full((1, 1, 20000), 0.4, np.float32))
        y = pr.sample(1, np.arange(20000), rng)[0]
        u = randomized_pit(pr.sample(400, np.arange(20000), rng), y, rng)
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float(((u > 0.1) & (u < 0.9)).mean()), 0.8, delta=0.02)

    def test_brasa_maps_predict(self) -> None:
        K = np.full((12, 2, 2), 4.0, np.float32)
        B = np.full((12, 2, 2), 0.5, np.float32)
        out = BrasaMaps(K, B).predict(np.array([[0, 1], [2, 0]], np.float32), 3)
        np.testing.assert_allclose(out, [[0.5, 4.0], [8.0, 0.5]])


@unittest.skipUnless(os.environ.get("BRASA_DERIVED"), "set BRASA_DERIVED to run data checks")
class TestData(unittest.TestCase):
    def test_legacy_model_on_real_stack(self) -> None:
        der = Path(os.environ["BRASA_DERIVED"]).expanduser()
        stack = next((p for p in (der / "ml" / "eval_all", der / "ml" / "overlap") if (p / "meta.json").exists()), None)
        legacy = REPO / "model" / "unet.pt"
        if stack is None or not legacy.exists():
            self.skipTest("no stack or model/unet.pt")
        s = Stack(stack)
        p = Predictor.load(legacy, "cpu", "old")
        t = s.usable([2019])[6]
        mu = p.predict_month(s, t, "cpu").mean
        v = np.nansum(s.target(t))
        self.assertLess(abs(mu.sum() / v - 1), 0.25, "legacy U-Net global total off by > 25%")


if __name__ == "__main__":
    unittest.main()

"""MODIS -> VIIRS(S-NPP)-equivalent harmonization: fitting + prediction.

Model (hierarchical ratio with seasonal modulation and a small-fire floor):
  V_hat[c,y,m] = k[c] * s[P1(c), m] * M[c,y,m]          if M > 0
               = b[P1(c), m]                             if M = 0 and c is an active cell
  k[c]   : per 0.25deg cell VIIRS/MODIS ratio, shrunk toward the 1deg parent,
           which is shrunk toward the 5deg parent, then the global ratio.
           shrinkage: k_child = (V + a*k_parent) / (M + a)   (a in MODIS-detection units)
  s[p,m] : 1deg x calendar-month modulation  = k_p,m / k_p  (also shrunk)
  b[p,m] : expected VIIRS count of an active 0.25deg cell in a month MODIS saw nothing
           (fires too small/brief for MODIS), per 1deg cell x calendar month.
"""
import numpy as np
import pandas as pd


def parents(cell):
    row, col = cell // 1440, cell % 1440
    return (row // 4) * 360 + col // 4, (row // 20) * 72 + col // 20


def _shrunk(V, M, prior, a):
    return (V + a * prior) / (M + a)


BINS = np.array([0, 1, 2, 5, 10, 25, 100, np.inf])


def fit(train, a=20.0, seasonal=True, floor=True, n_years=None, intensity=False):
    """train: DataFrame cell, year, month, M, V (one row per cell-month where M>0 or V>0)."""
    t = train.copy()
    t["p1"], t["p5"] = parents(t["cell"].to_numpy())
    kg = t.V.sum() / t.M.sum()
    pos = t[t.M > 0]
    g5 = pos.groupby("p5")[["V", "M"]].sum()
    k5 = _shrunk(g5.V, g5.M, kg, a)
    g1 = pos.groupby("p1")[["V", "M"]].sum()
    p5_of_p1 = pos.groupby("p1")["p5"].first()
    k1 = _shrunk(g1.V, g1.M, k5.reindex(p5_of_p1).to_numpy(), a)
    gc = pos.groupby("cell")[["V", "M"]].sum()
    p1_of_c = pos.groupby("cell")["p1"].first()
    kc = _shrunk(gc.V, gc.M, k1.reindex(p1_of_c).to_numpy(), a)
    model = dict(kg=kg, k5=k5, k1=k1, kc=kc, a=a)
    if seasonal:
        gm = pos.groupby(["p1", "month"])[["V", "M"]].sum()
        k1v = k1.reindex(gm.index.get_level_values(0)).to_numpy()
        model["s"] = _shrunk(gm.V, gm.M, k1v, a) / k1v
    if intensity:  # count-dependent correction: small MODIS counts hide relatively more small fires
        kk = ratio(model, pos.cell.to_numpy(), pos.month.to_numpy())
        b_ = np.digitize(pos.M.to_numpy(), BINS[1:-1], right=True)
        g = pd.DataFrame({"b": b_, "V": pos.V.to_numpy(), "P": kk * pos.M.to_numpy()}).groupby("b").sum()
        model["g"] = (g.V / g.P).reindex(range(len(BINS) - 1)).fillna(1.0).to_numpy()
    if floor:
        n_years = n_years or t.year.nunique()
        active = t.groupby("cell")["p1"].first()
        n_active = active.groupby(active).size()                    # active cells per 1deg
        z = t[t.M == 0].groupby(["p1", "month"])["V"].sum()          # V seen with M == 0
        pos_cm = pos.groupby(["p1", "month"]).size()                 # cell-months with M > 0
        idx = pd.MultiIndex.from_product([n_active.index, range(1, 13)], names=["p1", "month"])
        denom = n_active.reindex(idx.get_level_values(0)).to_numpy() * n_years \
            - pos_cm.reindex(idx).fillna(0).to_numpy()
        model["b"] = pd.Series(z.reindex(idx).fillna(0).to_numpy() / np.maximum(denom, 1), idx)
        model["active"] = active
    return model


def ratio(model, cell, month, M=None):
    """Per-row multiplicative factor for MODIS counts."""
    p1, p5 = parents(cell)
    k = model["kc"].reindex(cell).to_numpy()
    k = np.where(np.isnan(k), model["k1"].reindex(p1).to_numpy(), k)
    k = np.where(np.isnan(k), model["k5"].reindex(p5).to_numpy(), k)
    k = np.where(np.isnan(k), model["kg"], k)
    if "s" in model:
        s = model["s"].reindex(pd.MultiIndex.from_arrays([p1, month])).to_numpy()
        k = k * np.where(np.isnan(s), 1.0, s)
    if "g" in model and M is not None:
        k = k * model["g"][np.digitize(M, BINS[1:-1], right=True)]
    return k


def predict_regions(model, modis, region_of_cell):
    """Predict VIIRS-equivalent totals per (region, year, month).

    modis: cell, year, month, M (rows with M > 0)
    region_of_cell: function cell-array -> region-id array
    """
    m = modis[modis.M > 0].copy()
    m["pred"] = ratio(model, m.cell.to_numpy(), m.month.to_numpy(), m.M.to_numpy()) * m.M
    m["region"] = region_of_cell(m.cell.to_numpy())
    out = m.groupby(["region", "year", "month"])["pred"].sum()
    if "b" in model:
        act = model["active"]
        a = pd.DataFrame({"cell": act.index.to_numpy(), "p1": act.to_numpy()})
        a["region"] = region_of_cell(a.cell.to_numpy())
        # floor over ALL active cells, then remove the floor for cells where MODIS saw fire
        bt = model["b"].rename("b").reset_index()
        full = a.merge(bt, on="p1").groupby(["region", "month"])["b"].sum()
        m["p1"] = parents(m.cell.to_numpy())[0]
        m = m.merge(bt, on=["p1", "month"], how="left").fillna({"b": 0})
        used = m.groupby(["region", "year", "month"])["b"].sum()
        years = modis.year.unique()
        grid = pd.MultiIndex.from_product([full.index.get_level_values(0).unique(), years, range(1, 13)],
                                          names=["region", "year", "month"])
        fl = full.reindex(pd.MultiIndex.from_arrays(
            [grid.get_level_values(0), grid.get_level_values(2)])).to_numpy()
        fl = pd.Series(fl, grid).fillna(0)
        out = out.reindex(grid).fillna(0) + fl - used.reindex(grid).fillna(0)
    return out

"""Two-league model: Premier League + Championship, so promoted teams arrive with a rating.

Every (team, period) with matches is a CELL c; a period is a 91-day block inside one season (it never crosses a season, so a
team plays in one league per cell).  A cell has a league-RELATIVE attack A_c and defence B_c.  Match i, home cell ch, away cell ca,
l_i = 1 for a Championship match:
    log lam = iota + eta + A_ch - B_ca + g1 l + g2 l          log mu = iota + A_ca - B_ch + g1 l
(g1: Championship goal level, g2: Championship home-advantage shift).  Within one league only relative strengths matter, so what
links the two scales is the prior on a team's ABSOLUTE strength (Premier League scale, E0 offset = 0):
    sA_c = A_c + m_a f_c,    sB_c = B_c + m_d f_c        (f_c = 1 if the cell is a Championship cell; m < 0 = weaker league)
    penalty = k0 * sum_{first cell of each team} (A_c^2 + B_c^2)  +  k1 * sum_{c linked to previous cell c' of the same team} ((sA_c - sA_c')^2 + (sB_c - sB_c')^2)
         + tiny ridge on m.
A team promoted from the Championship keeps its absolute strength (up to the random-walk step), which is how m_a, m_d are learned.
Derivatives: with e_i = x_i - lam_i, f_i = y_i - mu_i, w_i weights, d_c = sA_c - sA_c' for a linked cell:
    dl/dA_c = sum_{i: ch=c} w e_i + sum_{i: ca=c} w f_i  - 2 k1 (d_c) [if c has a previous cell] + 2 k1 (d_next) [if c has a next cell] - 2 k0 A_c [first cell]
    dl/dB_c = -(sum_{i: ca=c} w e_i + sum_{i: ch=c} w f_i) with the same penalty terms on sB
    dl/dm_a = -2 k1 sum_linked d_c (f_c - f_c')   - 2 k_m m_a
    dl/dg1 = sum w (e + f) l,    dl/dg2 = sum w e l,    dl/diota = sum w (e + f),   dl/deta = sum w e.
A forecast for a Premier League match uses each team's LAST cell in absolute form, s = A + m f (the random-walk forecast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import poisson as P
from .walkforward import week_periods

K_M = 1e-3


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def season_year(dates):
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    return np.where(d.month >= 8, d.year, d.year - 1).astype(int)  # a season runs 1 Aug to 31 Jul (2019/20 ended on 26 Jul 2020)


def period_key(dates, length_days=91):
    """season_year * 100 + block number inside the season (blocks start on 1 August; never cross a season)."""
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    sy = season_year(d)
    start = np.array([np.datetime64(f"{y}-08-01", "D") for y in sy])
    blk = (np.asarray(d, dtype="datetime64[D]") - start).astype(int) // int(length_days)
    return sy * 100 + blk


def load_two_leagues(e0, e1_dir="data/raw_e1"):
    """e0: DataFrame from walkforward.load_matches (Premier League). Adds Championship rows. Returns (frame, team_names).

    The frame has columns Date, HomeTeam, AwayTeam, FTHG, FTAG, y, league (0/1), h2, a2 (ids over both leagues) and the
    original index for Premier League rows (Championship rows get indices from 10**7)."""
    from pathlib import Path
    from .data import parse_date
    a = e0[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "y"]].copy()
    a["league"] = 0
    frames = []
    for f in sorted(Path(e1_dir).glob("E1_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "FTR" not in df.columns:
            continue
        df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
        frames.append(pd.DataFrame({"Date": parse_date(df["Date"]), "HomeTeam": df["HomeTeam"], "AwayTeam": df["AwayTeam"],
                                    "FTHG": df["FTHG"].astype(float), "FTAG": df["FTAG"].astype(float),
                                    "y": df["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)}))
    if not frames:
        raise FileNotFoundError(f"no Championship files in {e1_dir}: run scripts/get_championship.sh")
    b = pd.concat(frames, ignore_index=True)
    b["league"] = 1
    b.index = b.index + 10 ** 7
    full = pd.concat([a, b]).sort_values("Date", kind="stable")
    teams = sorted(set(full["HomeTeam"]) | set(full["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    full["h2"] = full["HomeTeam"].map(idx).astype(int)
    full["a2"] = full["AwayTeam"].map(idx).astype(int)
    return full, teams


def build_cells(train, length_days=91):
    """Cell structure for a training frame: per-match home/away cell, and per-cell team, key, league flag, previous cell."""
    pk = period_key(train["Date"], length_days)
    keys = np.concatenate([np.stack([pk, train["h2"].values], 1), np.stack([pk, train["a2"].values], 1)])
    uniq = np.unique(keys, axis=0)  # sorted by (period key, team)
    cell_of = {(int(k), int(t)): i for i, (k, t) in enumerate(uniq)}
    ch = np.array([cell_of[(int(k), int(t))] for k, t in zip(pk, train["h2"].values)])
    ca = np.array([cell_of[(int(k), int(t))] for k, t in zip(pk, train["a2"].values)])
    n = len(uniq)
    flag = np.zeros(n)
    np.maximum.at(flag, ch, train["league"].values.astype(float))
    np.maximum.at(flag, ca, train["league"].values.astype(float))
    prev = np.full(n, -1)
    order = np.lexsort((uniq[:, 0], uniq[:, 1]))  # by team, then period
    for a_, b_ in zip(order[:-1], order[1:]):
        if uniq[a_, 1] == uniq[b_, 1]:
            prev[b_] = a_
    return {"ch": ch, "ca": ca, "keys": uniq, "team": uniq[:, 1], "flag": flag, "prev": prev, "n": n}


# ----------------------------------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------------------------------
NG = 6  # iota, eta, g1, g2, m_a, m_d


def loglik_and_grad(theta, cells, x, y, w, l, k0, k1):
    n = cells["n"]
    iota, eta, g1, g2, ma, md = theta[:NG]
    A, B = theta[NG: NG + n], theta[NG + n: NG + 2 * n]
    ch, ca, flag, prev = cells["ch"], cells["ca"], cells["flag"], cells["prev"]
    lam = np.exp(iota + eta + A[ch] - B[ca] + (g1 + g2) * l)
    mu = np.exp(iota + A[ca] - B[ch] + g1 * l)
    ll = np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu))
    e, f = w * (x - lam), w * (y - mu)
    gA = np.bincount(ch, weights=e, minlength=n) + np.bincount(ca, weights=f, minlength=n)
    gB = -(np.bincount(ca, weights=e, minlength=n) + np.bincount(ch, weights=f, minlength=n))
    g_iota, g_eta = np.sum(e + f), np.sum(e)
    g_g1, g_g2 = np.sum((e + f) * l), np.sum(e * l)
    first = prev < 0
    ll -= k0 * (np.sum(A[first] ** 2) + np.sum(B[first] ** 2))
    gA[first] -= 2 * k0 * A[first]
    gB[first] -= 2 * k0 * B[first]
    g_ma = g_md = 0.0
    lk = np.where(~first)[0]
    if len(lk):
        pv = prev[lk]
        for M_, gM, m_, g_m in ((A, gA, ma, "a"), (B, gB, md, "d")):
            s = M_ + m_ * flag
            d = s[lk] - s[pv]
            ll -= k1 * np.sum(d ** 2)
            np.add.at(gM, lk, -2 * k1 * d)
            np.add.at(gM, pv, 2 * k1 * d)
            g_m_val = -2 * k1 * np.sum(d * (flag[lk] - flag[pv]))
            if g_m == "a":
                g_ma = g_m_val
            else:
                g_md = g_m_val
    ll -= K_M * (ma ** 2 + md ** 2)
    g_ma -= 2 * K_M * ma
    g_md -= 2 * K_M * md
    return ll, np.concatenate([[g_iota, g_eta, g_g1, g_g2, g_ma, g_md], gA, gB])


def fit_cells(cells, x, y, w, l, k0, k1, theta0=None, max_iter=1000):
    n = cells["n"]
    if theta0 is None:
        theta0 = np.zeros(NG + 2 * n)
        theta0[0] = np.log(max(np.average(np.concatenate([x, y]), weights=np.concatenate([w, w])), 1e-3))
    f = lambda th: tuple(-v for v in loglik_and_grad(th, cells, x, y, w, l, k0, k1))
    r = minimize(f, np.array(theta0, float), jac=True, method="L-BFGS-B",
                 options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-12, "gtol": 1e-7})
    return r.x


def last_cells(cells):
    """Index of each team's last cell (latest period key)."""
    last = {}
    for c, (k, t) in enumerate(cells["keys"]):
        if int(t) not in last or k >= cells["keys"][last[int(t)]][0]:
            last[int(t)] = c
    return last


def warm_start(cells, theta_prev, cells_prev, T=None):
    """Initial theta for new cells: reuse the value of the same (period, team) cell if it existed, else the team's latest earlier cell."""
    n = cells["n"]
    th = np.zeros(NG + 2 * n)
    th[:NG] = theta_prev[:NG]
    npv = cells_prev["n"]
    A0, B0 = theta_prev[NG: NG + npv], theta_prev[NG + npv: NG + 2 * npv]
    old = {(int(k), int(t)): i for i, (k, t) in enumerate(cells_prev["keys"])}
    last_old = last_cells(cells_prev)
    for c, (k, t) in enumerate(cells["keys"]):
        j = old.get((int(k), int(t)))
        if j is None:
            j = last_old.get(int(t))
        if j is not None:
            th[NG + c], th[NG + n + c] = A0[j], B0[j]
    return th


# ----------------------------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------------------------
def walk_forward_two(full, start, end, k0, k1, xi=0.0, length_days=91):
    """Weekly walk-forward on Premier League matches; every earlier match of BOTH leagues is training data."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = full[(full["league"] == 0) & (full["Date"] >= start) & (full["Date"] < end)]
    theta, cells_prev, idx, probs = None, None, [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = full[full["Date"] < ref]
        cells = build_cells(train, length_days)
        w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi) if xi > 0 else np.ones(len(train))
        th0 = None if theta is None else warm_start(cells, theta, cells_prev)
        theta = fit_cells(cells, train["FTHG"].values.astype(float), train["FTAG"].values.astype(float), w,
                          train["league"].values.astype(float), k0, k1, th0)
        cells_prev = cells
        n = cells["n"]
        iota, eta, g1, g2, ma, md = theta[:NG]
        A, B = theta[NG: NG + n], theta[NG + n: NG + 2 * n]
        last = last_cells(cells)
        sa, sb = np.zeros(int(full[["h2", "a2"]].values.max()) + 1), np.zeros(int(full[["h2", "a2"]].values.max()) + 1)
        for t, c in last.items():
            sa[t], sb[t] = A[c] + ma * cells["flag"][c], B[c] + md * cells["flag"][c]
        h, a = block["h2"].values, block["a2"].values
        lam, mu = np.exp(iota + eta + sa[h] - sb[a]), np.exp(iota + sa[a] - sb[h])
        probs.append(P.outcome_probs(lam, mu))
        idx.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"])
    return out.sort_index(), theta


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    from pathlib import Path
    from .walkforward import load_matches, walk_forward
    from .drift import walk_forward_drift
    from . import metrics as M
    ap = argparse.ArgumentParser(description="Premier League + Championship joint model vs Model B and single-league drift.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--e1", default="data/raw_e1")
    ap.add_argument("--k1-grid", default="50,100,200")
    ap.add_argument("--k0", type=float, default=5.0)
    ap.add_argument("--out", default="reports/league2")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e0, teams = load_matches(args.data)
    full, names = load_two_leagues(e0, args.e1)
    both = sorted(set(e0["HomeTeam"]) & set(full.loc[full["league"] == 1, "HomeTeam"]))
    print(f"{len(names)} teams, {int((full['league']==1).sum())} Championship matches; clubs seen in both leagues: {len(both)}", flush=True)
    T = len(teams)
    base = {n: walk_forward(e0, T, s, e, 1.0, 5.0) for n, (s, e) in WINDOWS.items()}
    yy = {n: e0.loc[b.index, "y"].values for n, b in base.items()}
    newc = {n: (e0.loc[b.index, "h_new"].astype(bool) | e0.loc[b.index, "a_new"].astype(bool)).values for n, b in base.items()}
    one = {n: walk_forward_drift(e0, T, s, e, 5.0, 100.0, 0.0, 91) for n, (s, e) in WINDOWS.items()}

    def paired(new, old, y, mask=None):
        n_ = np.arange(len(y))
        d = -np.log(new.values[n_, y]) + np.log(old.values[n_, y])
        if mask is not None:
            d = d[mask]
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(len(d))

    rows = []
    s, e = WINDOWS["validation"]
    for k1 in [float(v) for v in args.k1_grid.split(",")]:
        wf, th = walk_forward_two(full, s, e, args.k0, k1)
        d, se, _ = paired(wf, base["validation"], yy["validation"])
        rows.append({"k1": k1, "LL": M.log_loss(wf.values, yy["validation"]), "delta_vs_B": d, "se": se, "m_a_last": th[4], "m_d_last": th[5]})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(out / "grid_validation.csv", index=False)
    best = g.loc[g["LL"].idxmin()]
    print("best on validation:", dict(best), flush=True)
    res = []
    for n, (s, e) in WINDOWS.items():
        wf, th = walk_forward_two(full, s, e, args.k0, best["k1"])
        for label, ref in (("Model B", base[n]), ("single-league drift", one[n])):
            d, se, nn = paired(wf, ref, yy[n])
            dn, sen, nnn = paired(wf, ref, yy[n], newc[n])
            res.append({"window": n, "vs": label, "n": nn, "LL_two": M.log_loss(wf.values, yy[n]), "delta": d, "se": se,
                        "n_promoted_matches": nnn, "delta_promoted": dn, "se_promoted": sen, "k1": best["k1"], "m_a": th[4], "m_d": th[5]})
            print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(out / "league2_results.csv", index=False)


if __name__ == "__main__":
    main()

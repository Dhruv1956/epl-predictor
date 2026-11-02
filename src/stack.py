"""Stacking: blend the shots-Poisson forecast with a completely different model (Elo + ordered logit).

Elo (results only, played in date order).  Every team starts at 1500.  Before a match
    d = R_home - R_away,   E_home = 1 / (1 + 10^(-(d + H) / 400))     (H = 65 fixed home-advantage points)
and afterwards, with S_home = 1 / 0.5 / 0 for a home win / draw / away win,
    R_home += K (S_home - E_home),   R_away -= K (S_home - E_home).
When more than 30 days pass without a match (the summer break) every rating is pulled back:
    R <- 1500 + carry * (R - 1500).
The pre-match d of a match depends only on earlier matches, so it is available before kickoff.

Ordered logit link (maps d to home/draw/away probabilities).  z = beta * d / 400, cutpoints c1 < c2 = c1 + exp(u):
    P(A) = F(c1 - z),   P(H) = 1 - F(c2 - z),   P(D) = F(c2 - z) - F(c1 - z),   F = logistic CDF.
With s1 = F(c1 - z), s2 = F(c2 - z), f_j = s_j (1 - s_j) the derivatives are
    dP(A)/dc1 = f1,  dP(A)/dz = -f1;   dP(H)/dc2 = -f2,  dP(H)/dz = f2;   dP(D)/dc1 = -f1, dP(D)/dc2 = f2, dP(D)/dz = f1 - f2.
The log-likelihood sum_i log P(y_i) is maximised over (beta, c1, u) on the training matches only.

Blending two forecasts p1 (shots Poisson) and p2 (Elo):
    linear pool   p = (1 - w) p1 + w p2
    log pool      q_k proportional to exp(a log p1_k + b log p2_k);   dL/da = sum (log p1_y - sum_k q_k log p1_k), same for b.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .walkforward import week_periods

HOME_ELO = 65.0
GAP_DAYS = 30


# ----------------------------------------------------------------------------------------------
# Elo
# ----------------------------------------------------------------------------------------------
def elo_prematch(df, K, carry, home=HOME_ELO, gap_days=GAP_DAYS):
    """Pre-match rating difference d = R_home - R_away for every row of df (sorted by date).
    Rows without a result (fixtures) get a d but do not update the ratings."""
    n_teams = int(max(df["h"].max(), df["a"].max())) + 1
    R = np.full(n_teams, 1500.0)
    d = np.zeros(len(df))
    h, a, y = df["h"].values, df["a"].values, df["y"].values
    played = df["FTHG"].notna().values & df["FTAG"].notna().values
    dates = df["Date"].values
    prev = None
    for i in range(len(df)):
        if prev is not None and (dates[i] - prev) > np.timedelta64(gap_days, "D"):
            R = 1500.0 + carry * (R - 1500.0)
        prev = dates[i]
        d[i] = R[h[i]] - R[a[i]]
        if played[i]:
            E = 1.0 / (1.0 + 10.0 ** (-(d[i] + home) / 400.0))
            S = 1.0 if y[i] == 0 else (0.5 if y[i] == 1 else 0.0)
            R[h[i]] += K * (S - E)
            R[a[i]] -= K * (S - E)
    return d


# ----------------------------------------------------------------------------------------------
# Ordered logit
# ----------------------------------------------------------------------------------------------
def _F(t):
    return 1.0 / (1.0 + np.exp(-t))


def ol_probs(params, d):
    beta, c1, u = params
    c2 = c1 + np.exp(u)
    z = beta * np.asarray(d, float) / 400.0
    s1, s2 = _F(c1 - z), _F(c2 - z)
    return np.column_stack([1.0 - s2, s2 - s1, s1])  # H, D, A


def ol_loglik_and_grad(params, d, y, w=None):
    beta, c1, u = params
    d, y = np.asarray(d, float), np.asarray(y, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    c2 = c1 + np.exp(u)
    z = beta * d / 400.0
    s1, s2 = _F(c1 - z), _F(c2 - z)
    f1, f2 = s1 * (1 - s1), s2 * (1 - s2)
    pH, pD, pA = 1.0 - s2, s2 - s1, s1
    P_ = np.column_stack([pH, pD, pA])
    n = np.arange(len(y))
    p_y = np.clip(P_[n, y], 1e-12, None)
    ll = np.sum(w * np.log(p_y))
    # derivatives of the three probabilities with respect to (z, c1, c2)
    dz = np.column_stack([f2, f1 - f2, -f1])
    dc1 = np.column_stack([np.zeros(len(y)), -f1, f1])
    dc2 = np.column_stack([-f2, f2, np.zeros(len(y))])
    gz, g1, g2 = (M[n, y] / p_y * w for M in (dz, dc1, dc2))
    g_beta = np.sum(gz * d / 400.0)
    g_c1 = np.sum(g1 + g2)
    g_u = np.sum(g2) * np.exp(u)
    return ll, np.array([g_beta, g_c1, g_u])


def fit_ol(d, y, w=None, theta0=None):
    f = lambda th: tuple(-v for v in ol_loglik_and_grad(th, d, y, w))
    r = minimize(f, np.array([1.0, -0.5, 0.0] if theta0 is None else theta0, float), jac=True, method="L-BFGS-B",
                 options={"ftol": 1e-13, "gtol": 1e-8})
    return r.x


def walk_forward_elo(df, d, start, end):
    """Weekly walk-forward: ordered-logit link refitted on matches strictly before each block, using their pre-match d."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dser = pd.Series(d, index=df.index)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta, idx, probs = None, [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = df[(df["Date"] < ref) & df["FTHG"].notna()]
        theta = fit_ol(dser.loc[train.index].values, train["y"].values, theta0=theta)
        probs.append(ol_probs(theta, dser.loc[block.index].values))
        idx.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"])
    return out.sort_index()


# ----------------------------------------------------------------------------------------------
# Blending
# ----------------------------------------------------------------------------------------------
def linear_pool(p1, p2, w):
    return (1.0 - w) * np.asarray(p1, float) + w * np.asarray(p2, float)


def log_pool(p1, p2, a, b):
    s1, s2 = np.log(np.clip(p1, 1e-12, 1)), np.log(np.clip(p2, 1e-12, 1))
    z = a * s1 + b * s2
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_pool_loglik_and_grad(params, p1, p2, y):
    a, b = params
    s1, s2 = np.log(np.clip(p1, 1e-12, 1)), np.log(np.clip(p2, 1e-12, 1))
    q = log_pool(p1, p2, a, b)
    n = np.arange(len(y))
    ll = np.sum(np.log(np.clip(q[n, y], 1e-15, 1)))
    return ll, np.array([np.sum(s1[n, y] - (q * s1).sum(1)), np.sum(s2[n, y] - (q * s2).sum(1))])


def fit_log_pool(p1, p2, y):
    f = lambda th: tuple(-v for v in log_pool_loglik_and_grad(th, p1, p2, y))
    return minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B").x


def per_match_ll(p, y):
    return -np.log(np.clip(np.asarray(p)[np.arange(len(y)), y], 1e-15, 1))


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    from pathlib import Path
    from .live import MODELS
    from .walkforward import load_matches, walk_forward
    from . import metrics as M
    ap = argparse.ArgumentParser(description="Stack the v2-shots Poisson forecast with an Elo model.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--k-grid", default="10,20,30,40")
    ap.add_argument("--carry-grid", default="1.0,0.75")
    ap.add_argument("--out", default="reports/stack")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)
    cfg = MODELS["v2-shots"]
    p1 = {n: walk_forward(df, T, s, e, cfg["xi"], cfg["kappa"], spec=cfg["spec"]) for n, (s, e) in WINDOWS.items()}
    yy = {n: df.loc[b.index, "y"].values for n, b in p1.items()}
    # --- Elo settings chosen on validation by the Elo model's own log loss
    rows = []
    s, e = WINDOWS["validation"]
    for K in [float(v) for v in args.k_grid.split(",")]:
        for carry in [float(v) for v in args.carry_grid.split(",")]:
            d = elo_prematch(df, K, carry)
            wf = walk_forward_elo(df, d, s, e)
            rows.append({"K": K, "carry": carry, "LL_elo": M.log_loss(wf.values, yy["validation"])})
            print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(out / "elo_grid_validation.csv", index=False)
    best = g.loc[g["LL_elo"].idxmin()]
    print("best Elo:", dict(best), flush=True)
    d = elo_prematch(df, best["K"], best["carry"])
    p2 = {n: walk_forward_elo(df, d, s, e) for n, (s, e) in WINDOWS.items()}
    # --- blend parameters fitted on validation only
    pv1, pv2, yv = p1["validation"].values, p2["validation"].values, yy["validation"]
    ws = np.round(np.arange(0, 1.01, 0.1), 2)
    lin = pd.DataFrame({"w": ws, "LL": [M.log_loss(linear_pool(pv1, pv2, w), yv) for w in ws]})
    lin.to_csv(out / "linear_pool_validation.csv", index=False)
    wbest = float(lin.loc[lin["LL"].idxmin(), "w"])
    ab = fit_log_pool(pv1, pv2, yv)
    print("linear pool best w =", wbest, " log pool a,b =", ab, flush=True)
    # ---- how the linear pool behaves in every window (diagnostic) and a choice from the two windows before the test
    for n in WINDOWS:
        pd.DataFrame(np.column_stack([p1[n].values, p2[n].values, yy[n]]),
                     columns=["p1H", "p1D", "p1A", "p2H", "p2D", "p2A", "y"], index=p1[n].index).to_csv(out / f"preds_{n}.csv")
    curve = pd.DataFrame({"w": ws})
    for n in WINDOWS:
        curve[n] = [M.log_loss(linear_pool(p1[n].values, p2[n].values, w), yy[n]) for w in ws]
    pre = np.vstack([p1["early"].values, p1["validation"].values]), np.vstack([p2["early"].values, p2["validation"].values])
    ypre = np.concatenate([yy["early"], yy["validation"]])
    curve["early+validation"] = [M.log_loss(linear_pool(pre[0], pre[1], w), ypre) for w in ws]
    curve.to_csv(out / "linear_pool_by_window.csv", index=False)
    print(curve.round(5).to_string(index=False), flush=True)
    wpre = float(curve.loc[curve["early+validation"].idxmin(), "w"])
    print("w chosen on early+validation:", wpre, flush=True)
    res = []
    for n in WINDOWS:
        y = yy[n]
        base = per_match_ll(p1[n].values, y)
        for name, p in (("elo alone", p2[n].values), ("linear pool", linear_pool(p1[n].values, p2[n].values, wbest)),
                        ("log pool", log_pool(p1[n].values, p2[n].values, *ab)),
                        ("linear pool, w from early+validation", linear_pool(p1[n].values, p2[n].values, wpre))):
            dl = per_match_ll(p, y) - base
            res.append({"window": n, "n": len(y), "blend": name, "LL_v2": base.mean(), "LL_blend": per_match_ll(p, y).mean(),
                        "delta": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)), "w": wbest, "a": ab[0], "b": ab[1],
                        "K": best["K"], "carry": best["carry"]})
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
    pd.DataFrame(res).to_csv(out / "stack_results.csv", index=False)


if __name__ == "__main__":
    main()

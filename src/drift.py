"""Drifting team strengths (random-walk prior), fitted from scratch.

Model B has ONE attack alpha_t and ONE defence beta_t per team, and copes with change by down-weighting old matches
(w = exp(-xi * age)). Here each team instead gets a strength PER PERIOD p (a block of L days):

    log lam = iota + eta + alpha[p,h] - beta[p,a]          log mu = iota + alpha[p,a] - beta[p,h]

and the strengths are tied together by a random-walk prior alpha[p] = alpha[p-1] + noise:

    ell = sum_i w_i ( x_i log lam_i - lam_i + y_i log mu_i - mu_i )
          - k0 * sum_t ( alpha[0,t]^2 + beta[0,t]^2 )
          - k1 * sum_{p>=1} sum_t ( (alpha[p,t]-alpha[p-1,t])^2 + (beta[p,t]-beta[p-1,t])^2 )

k1 is the stiffness: k1 -> infinity gives constant strengths, small k1 lets them drift. The prior variance of one
step is 1/(2 k1). Gradient, with e_i = x_i - lam_i, f_i = y_i - mu_i and p_i the period of match i:
    d ell/d alpha[p,t] = sum_{i: p_i=p} w_i ( e_i 1[h_i=t] + f_i 1[a_i=t] )  - penalty terms
    d ell/d beta[p,t]  = - sum_{i: p_i=p} w_i ( e_i 1[a_i=t] + f_i 1[h_i=t] ) - penalty terms
    penalty term for alpha[p]: -2 k0 alpha[0] (p=0);  -2 k1 (alpha[p]-alpha[p-1]) + 2 k1 (alpha[p+1]-alpha[p])
A forecast for a future match uses the LAST period's strengths (the random-walk forecast is the last value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import poisson as P
from .walkforward import week_periods


def period_index(dates, epoch, length_days):
    """Calendar block number of each date since a fixed epoch, then compressed to 0..n-1 over the blocks present."""
    days = (np.asarray(pd.to_datetime(dates), dtype="datetime64[D]") - np.datetime64(pd.Timestamp(epoch), "D")).astype(int)
    d = days // int(length_days)
    _, inv = np.unique(d, return_inverse=True)
    return inv


def unpack(theta, T, nP):
    iota, eta = theta[0], theta[1]
    A = theta[2: 2 + nP * T].reshape(nP, T)
    B = theta[2 + nP * T: 2 + 2 * nP * T].reshape(nP, T)
    return iota, eta, A, B


def loglik_and_grad(theta, h, a, x, y, w, p, T, nP, k0, k1):
    iota, eta, A, B = unpack(theta, T, nP)
    lam = np.exp(iota + eta + A[p, h] - B[p, a])
    mu = np.exp(iota + A[p, a] - B[p, h])
    ll = np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu))
    e, f = w * (x - lam), w * (y - mu)
    flat = lambda t, v: np.bincount(p * T + t, weights=v, minlength=nP * T)
    gA = (flat(h, e) + flat(a, f)).reshape(nP, T)
    gB = -(flat(a, e) + flat(h, f)).reshape(nP, T)
    g_iota = np.sum(e + f)
    g_eta = np.sum(e)
    for M_, gM in ((A, gA), (B, gB)):
        ll -= k0 * np.sum(M_[0] ** 2)
        gM[0] -= 2 * k0 * M_[0]
        if nP > 1:
            d = M_[1:] - M_[:-1]
            ll -= k1 * np.sum(d ** 2)
            gM[1:] -= 2 * k1 * d
            gM[:-1] += 2 * k1 * d
    return ll, np.concatenate([[g_iota, g_eta], gA.ravel(), gB.ravel()])


def fit_drift(h, a, x, y, w, p, T, k0, k1, theta0=None, max_iter=1000):
    h, a, p = np.asarray(h), np.asarray(a), np.asarray(p)
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    nP = int(p.max()) + 1
    if theta0 is None:
        theta0 = np.zeros(2 + 2 * nP * T)
        theta0[0] = np.log(max(np.average(np.concatenate([x, y]), weights=np.concatenate([w, w])), 1e-3))
    f = lambda th: tuple(-v for v in loglik_and_grad(th, h, a, x, y, w, p, T, nP, k0, k1))
    r = minimize(f, np.array(theta0, float), jac=True, method="L-BFGS-B",
                 options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-12, "gtol": 1e-7})
    return r.x, nP


def _extend(theta, T, nP_old, nP_new):
    """Warm start: new periods start at the previous last period's strengths."""
    iota, eta, A, B = unpack(theta, T, nP_old)
    A2 = np.vstack([A, np.repeat(A[-1:], nP_new - nP_old, axis=0)])
    B2 = np.vstack([B, np.repeat(B[-1:], nP_new - nP_old, axis=0)])
    return np.concatenate([[iota, eta], A2.ravel(), B2.ravel()])


def walk_forward_drift(df, T, start, end, k0, k1, xi=0.0, length_days=91):
    """Weekly walk-forward: refit on matches strictly before each block, forecast with the last period's strengths."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    epoch = df["Date"].min()
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta, nP_prev, idx, probs = None, 0, [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = df[df["Date"] < ref]
        p = period_index(train["Date"], epoch, length_days)
        nP = int(p.max()) + 1
        w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi) if xi > 0 else np.ones(len(train))
        th0 = None if theta is None else (theta if nP == nP_prev else _extend(theta, T, nP_prev, nP))
        theta, nP = fit_drift(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, p, T,
                              k0, k1, theta0=th0)
        nP_prev = nP
        iota, eta, A, B = unpack(theta, T, nP)
        h, a = block["h"].values, block["a"].values
        lam = np.exp(iota + eta + A[-1, h] - B[-1, a])
        mu = np.exp(iota + A[-1, a] - B[-1, h])
        probs.append(P.outcome_probs(lam, mu))
        idx.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"])
    return out.sort_index()


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def _paired(new, old, y):
    n = np.arange(len(y))
    d = -np.log(new.values[n, y]) + np.log(old.values[n, y])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main(argv=None):
    import argparse
    from pathlib import Path
    from .walkforward import load_matches, walk_forward
    from . import metrics as M
    ap = argparse.ArgumentParser(description="Drifting (random-walk) team strengths vs Model B.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--xi", type=float, default=1.0, help="Model B decay")
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--k1-grid", default="3,10,30,100")
    ap.add_argument("--xi-grid", default="0,0.5")
    ap.add_argument("--length", type=int, default=91)
    ap.add_argument("--out", default="reports/drift")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)
    base = {n: walk_forward(df, T, s, e, args.xi, args.kappa) for n, (s, e) in WINDOWS.items()}
    yy = {n: df.loc[b.index, "y"].values for n, b in base.items()}
    rows = []
    s, e = WINDOWS["validation"]
    for xi in [float(v) for v in args.xi_grid.split(",")]:
        for k1 in [float(v) for v in args.k1_grid.split(",")]:
            wf = walk_forward_drift(df, T, s, e, args.kappa, k1, xi, args.length)
            d, se = _paired(wf, base["validation"], yy["validation"])
            rows.append({"xi": xi, "k1": k1, "LL": M.log_loss(wf.values, yy["validation"]), "delta": d, "se": se})
            print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(out / "grid_validation.csv", index=False)
    best = g.loc[g["LL"].idxmin()]
    print("best on validation:", dict(best), flush=True)
    res = []
    for n, (s, e) in WINDOWS.items():
        wf = walk_forward_drift(df, T, s, e, args.kappa, best["k1"], best["xi"], args.length)
        d, se = _paired(wf, base[n], yy[n])
        res.append({"window": n, "n": len(yy[n]), "LL_B": M.log_loss(base[n].values, yy[n]), "LL_drift": M.log_loss(wf.values, yy[n]),
                    "delta": d, "se": se, "k1": best["k1"], "xi": best["xi"]})
        print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(out / "drift_results.csv", index=False)


if __name__ == "__main__":
    main()

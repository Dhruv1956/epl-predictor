"""Two cheap improvements to Model B, tested the project's way (validation chooses, early window and test check).

1. Recalibration of the outcome probabilities (src.improve.recalibrate_*):
       z_k = t * log p_k + b_k        (b_A = 0),   p'_k = exp(z_k) / sum_j exp(z_j)
   t is a temperature (t < 1 flattens, t > 1 sharpens), b_H and b_D shift home and draw probability.
   The 3 parameters are fitted on the VALIDATION window only and then applied unchanged to the other windows.
   Log-likelihood  L(t, b) = sum_i log p'_{i, y_i}.  With s_{ik} = log p_ik and p' the recalibrated probabilities:
       dL/dt   = sum_i ( s_{i, y_i} - sum_k p'_ik s_ik )
       dL/db_k = sum_i ( 1[y_i = k] - p'_ik )          for k in {H, D}
   (the usual softmax-regression gradient: observed minus expected).

2. Separate ridge strengths for attack and defence (fit_split, walk_forward_split):
       ell(theta) = loglik(theta) - kappa_a * sum alpha_t^2 - kappa_b * sum beta_t^2
   Model B uses one kappa for both. The gradient is Model B's unpenalised gradient minus 2 kappa_a alpha_t and
   2 kappa_b beta_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import metrics as M
from . import poisson as P
from .walkforward import week_periods

OUT = ["H", "D", "A"]


# ----------------------------------------------------------------------------------------------
# 1. Calibration
# ----------------------------------------------------------------------------------------------
def recalibrate(probs, params):
    """Apply (t, bH, bD) to probabilities (n, 3). Returns (n, 3)."""
    t, bH, bD = params
    s = np.log(np.clip(np.asarray(probs, float), 1e-12, 1.0))
    z = t * s + np.array([bH, bD, 0.0])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def recal_loglik_and_grad(params, probs, y):
    probs = np.asarray(probs, float)
    y = np.asarray(y, int)
    s = np.log(np.clip(probs, 1e-12, 1.0))
    q = recalibrate(probs, params)
    n = len(y)
    ll = np.sum(np.log(np.clip(q[np.arange(n), y], 1e-15, 1.0)))
    d_t = np.sum(s[np.arange(n), y] - (q * s).sum(axis=1))
    onehot = np.eye(3)[y]
    d_b = (onehot - q).sum(axis=0)[:2]
    return ll, np.array([d_t, d_b[0], d_b[1]])


def fit_recalibration(probs, y):
    """Maximum-likelihood (t, bH, bD) on a set of forecasts. Returns the parameter vector."""
    f = lambda th: tuple(-v for v in recal_loglik_and_grad(th, probs, y))
    r = minimize(f, np.array([1.0, 0.0, 0.0]), jac=True, method="L-BFGS-B")
    return r.x


def reliability(probs, y, bins=5):
    """Equal-count bins per outcome: mean forecast, observed frequency, count, standard error."""
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    rows = []
    for k, name in enumerate(OUT):
        order = np.argsort(probs[:, k])
        for b, idx in enumerate(np.array_split(order, bins)):
            p = probs[idx, k].mean()
            o = float((y[idx] == k).mean())
            rows.append({"outcome": name, "bin": b + 1, "n": len(idx), "mean_forecast": p, "observed": o,
                         "se": float(np.sqrt(max(p * (1 - p), 1e-9) / len(idx)))})
    return pd.DataFrame(rows)


def paired(ll_new, ll_old):
    d = np.asarray(ll_new) - np.asarray(ll_old)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def per_match_ll(probs, y):
    probs = np.asarray(probs, float)
    return -np.log(np.clip(probs[np.arange(len(y)), np.asarray(y, int)], 1e-15, 1.0))


# ----------------------------------------------------------------------------------------------
# 2. Separate kappa for attack and defence
# ----------------------------------------------------------------------------------------------
def fit_split(h, a, x, y, w, T, kappa_a, kappa_b, theta0=None, max_iter=1000):
    h, a = np.asarray(h), np.asarray(a)
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    if theta0 is None:
        theta0 = np.zeros(P.n_params(T, 0, False, False))
        theta0[0] = np.log(max(np.average(np.concatenate([x, y]), weights=np.concatenate([w, w])), 1e-3))

    def neg(theta):
        ll, g = P.loglik_and_grad(theta, h, a, x, y, w, T, 0.0)  # unpenalised
        _, _, alpha, beta = P.unpack(theta, T)
        ll -= kappa_a * (alpha @ alpha) + kappa_b * (beta @ beta)
        g = g.copy()
        g[2: 2 + T] -= 2 * kappa_a * alpha
        g[2 + T: 2 + 2 * T] -= 2 * kappa_b * beta
        return -ll, -g

    r = minimize(neg, np.array(theta0, float), jac=True, method="L-BFGS-B",
                 options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-13, "gtol": 1e-8})
    return r.x


def walk_forward_split(df, T, start, end, xi, kappa_a, kappa_b):
    """Weekly walk-forward (fit on matches strictly before each block) with separate attack and defence ridge."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta, idx, probs = None, [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = df[df["Date"] < ref]
        w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi)
        theta = fit_split(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, T,
                          kappa_a, kappa_b, theta0=theta)
        lam, mu = P.rates(theta, block["h"].values, block["a"].values, T)
        probs.append(P.outcome_probs(lam, mu))
        idx.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"])
    return out.sort_index()


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    from pathlib import Path
    from .walkforward import load_matches, walk_forward
    ap = argparse.ArgumentParser(description="Calibration and split-kappa experiments (Model B base).")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--grid", default="2,3.5,5,8,12,20", help="kappa values tried for attack and for defence")
    ap.add_argument("--out", default="reports/improve")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)

    # ---- baseline walk-forward predictions in all three windows
    base = {n: walk_forward(df, T, s, e, args.xi, args.kappa) for n, (s, e) in WINDOWS.items()}
    yy = {n: df.loc[b.index, "y"].values for n, b in base.items()}

    # ---- 1. calibration
    rel = pd.concat([reliability(base[n].values, yy[n]).assign(window=n) for n in WINDOWS])
    rel.to_csv(out / "reliability.csv", index=False)
    params = fit_recalibration(base["validation"].values, yy["validation"])
    print("recalibration fitted on validation: t=%.4f bH=%.4f bD=%.4f" % tuple(params))
    rows = []
    for n in WINDOWS:
        q = recalibrate(base[n].values, params)
        d, se = paired(per_match_ll(q, yy[n]), per_match_ll(base[n].values, yy[n]))
        own = fit_recalibration(base[n].values, yy[n])
        rows.append({"window": n, "n": len(yy[n]), "LL_base": M.log_loss(base[n].values, yy[n]), "LL_recal": M.log_loss(q, yy[n]),
                     "delta": d, "se": se, "Brier_base": M.brier_score(base[n].values, yy[n]), "Brier_recal": M.brier_score(q, yy[n]),
                     "draw_pred": float(base[n].values[:, 1].mean()), "draw_act": float((yy[n] == 1).mean()),
                     "own_t": own[0], "own_bH": own[1], "own_bD": own[2]})
    pd.DataFrame(rows).to_csv(out / "recalibration.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---- 2. split kappa: grid on validation, then evaluate the winner on the other windows
    ks = [float(v) for v in args.grid.split(",")]
    grid = []
    s, e = WINDOWS["validation"]
    for ka in ks:
        for kb in ks:
            wf = walk_forward_split(df, T, s, e, args.xi, ka, kb)
            grid.append({"kappa_a": ka, "kappa_b": kb, "LL": M.log_loss(wf.values, yy["validation"])})
    g = pd.DataFrame(grid)
    g.to_csv(out / "kappa_grid_validation.csv", index=False)
    best = g.loc[g["LL"].idxmin()]
    print("validation grid (rows kappa_a, cols kappa_b):")
    print(g.pivot(index="kappa_a", columns="kappa_b", values="LL").round(5).to_string())
    print("best:", dict(best))
    rows = []
    for n, (s, e) in WINDOWS.items():
        wf = walk_forward_split(df, T, s, e, args.xi, best["kappa_a"], best["kappa_b"])
        d, se = paired(per_match_ll(wf.values, yy[n]), per_match_ll(base[n].values, yy[n]))
        rows.append({"window": n, "n": len(yy[n]), "LL_base": M.log_loss(base[n].values, yy[n]), "LL_split": M.log_loss(wf.values, yy[n]),
                     "delta": d, "se": se, "kappa_a": best["kappa_a"], "kappa_b": best["kappa_b"]})
    pd.DataFrame(rows).to_csv(out / "split_kappa.csv", index=False)
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()

"""Bivariate Poisson goals model (Karlis and Ntzoufras, 2003), fitted from scratch.

Model B treats home goals X and away goals Y as independent Poisson counts. Here they share a common
"tempo" count Z:

    X = X1 + Z,   Y = Y1 + Z,   X1 ~ Pois(lam1),  Y1 ~ Pois(lam2),  Z ~ Pois(lam3),  all independent.

    log lam1 = iota + eta + alpha[h] - beta[a]        (same linear predictors as Model B)
    log lam2 = iota + alpha[a] - beta[h]
    lam3     = exp(g)                                  (one constant, so Cov(X, Y) = lam3 >= 0)

Marginals are X ~ Pois(lam1 + lam3) and Y ~ Pois(lam2 + lam3). Summing over the shared part Z = z gives

    P(x, y) = exp(-(lam1+lam2+lam3)) lam1^x / x!  lam2^y / y!  *  S,
    S = sum_{k=0}^{min(x,y)} C(x,k) C(y,k) k! r^k,      r = lam3 / (lam1 lam2).

Objective (same weights and ridge penalty as Model B):

    ell = sum_i w_i log P(x_i, y_i)  -  kappa * (sum alpha^2 + sum beta^2).

Gradient, derived by hand. Let pi_k be the k-th term of S divided by S (a distribution over k) and
K_i = sum_k k pi_k, which is E[Z | x_i, y_i]. Since r^k depends on lam1, lam2, lam3 only through
r = lam3 / (lam1 lam2), d log S / d log lam1 = d log S / d log lam2 = -K and d log S / d log lam3 = +K. So

    d ell_i / d log lam1 = x_i - lam1 - K_i  =: e_i
    d ell_i / d log lam2 = y_i - lam2 - K_i  =: f_i
    d ell_i / d g        = K_i - lam3

and the chain rule through the linear predictors gives exactly the Model B formulas with e, f in place
of the plain residuals x - lam, y - mu:

    d ell / d iota    = sum w (e + f)
    d ell / d eta     = sum w e
    d ell / d alpha_t = sum_{h=t} w e + sum_{a=t} w f - 2 kappa alpha_t
    d ell / d beta_t  = -sum_{a=t} w e - sum_{h=t} w f - 2 kappa beta_t
    d ell / d g       = sum w (K - lam3)

(Interpretation: K is the part of the observed goals that the shared component Z is believed to explain.)
When lam3 -> 0, K -> 0 and everything reduces to Model B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from . import metrics as M
from . import poisson as P
from .walkforward import week_periods

G_BOUNDS = (np.log(1e-4), np.log(0.8))  # lam3 between 0.0001 and 0.8


def n_params_bp(T):
    return 2 + 2 * T + 1


def _split(theta, T):
    return theta[: 2 + 2 * T], theta[2 + 2 * T]


class _Data:
    """Per-training-set constants, so each likelihood evaluation only does cheap array work."""

    def __init__(self, x, y):
        self.x, self.y = x.astype(float), y.astype(float)
        mn = np.minimum(x, y).astype(int)
        self.kmax = int(mn.max()) if len(mn) else 0
        k = np.arange(self.kmax + 1)[None, :]
        xx, yy = self.x[:, None], self.y[:, None]
        with np.errstate(invalid="ignore"):
            lb = (gammaln(xx + 1) - gammaln(k + 1) - gammaln(xx - k + 1)
                  + gammaln(yy + 1) - gammaln(k + 1) - gammaln(yy - k + 1) + gammaln(k + 1))
        lb = np.where(k <= mn[:, None], lb, -np.inf)
        self.lb, self.k = lb, k.astype(float)
        self.const = gammaln(self.x + 1) + gammaln(self.y + 1)


def _terms(lam1, lam2, lam3, D):
    """Per-match log pmf and K = E[Z | x, y]."""
    logr = np.log(lam3) - np.log(lam1) - np.log(lam2)
    A = D.lb + D.k * logr[:, None]
    logS = logsumexp(A, axis=1)
    K = (np.exp(A - logS[:, None]) * D.k).sum(axis=1)
    logp = -(lam1 + lam2 + lam3) + D.x * np.log(lam1) + D.y * np.log(lam2) - D.const + logS
    return logp, K


def loglik_and_grad_bp(theta, h, a, x, y, w, T, kappa, D=None):
    D = D or _Data(x, y)
    base, g = _split(theta, T)
    log1, log2 = P.linear_predictors(base, h, a, T)
    lam1, lam2, lam3 = np.exp(log1), np.exp(log2), np.exp(g)
    logp, K = _terms(lam1, lam2, lam3, D)
    _, _, alpha, beta = P.unpack(base, T)
    ll = np.sum(w * logp) - kappa * (alpha @ alpha + beta @ beta)
    e, f = D.x - lam1 - K, D.y - lam2 - K
    grad = np.zeros_like(theta, dtype=float)
    grad[0] = np.sum(w * (e + f))
    grad[1] = np.sum(w * e)
    grad[2: 2 + T] = np.bincount(h, weights=w * e, minlength=T) + np.bincount(a, weights=w * f, minlength=T) - 2 * kappa * alpha
    grad[2 + T: 2 + 2 * T] = -np.bincount(a, weights=w * e, minlength=T) - np.bincount(h, weights=w * f, minlength=T) - 2 * kappa * beta
    grad[2 + 2 * T] = np.sum(w * (K - lam3))
    return ll, grad


@dataclass
class BPFit:
    theta: np.ndarray
    loglik: float
    success: bool
    n_iter: int

    @property
    def lam3(self):
        return float(np.exp(self.theta[-1]))


def fit_bp(h, a, x, y, w, T, kappa, theta0=None, max_iter=1000):
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    h, a = np.asarray(h), np.asarray(a)
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    D = _Data(x, y)
    if theta0 is None:
        theta0 = np.zeros(n_params_bp(T))
        theta0[0] = np.log(max(np.average(np.concatenate([x, y]), weights=np.concatenate([w, w])), 1e-3))
        theta0[-1] = np.log(0.05)
    bounds = [(None, None)] * (len(theta0) - 1) + [G_BOUNDS]

    def neg(theta):
        ll, gr = loglik_and_grad_bp(theta, h, a, x, y, w, T, kappa, D)
        return -ll, -gr

    r = minimize(neg, np.array(theta0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-13, "gtol": 1e-8})
    return BPFit(r.x, -r.fun, bool(r.success), int(r.nit))


def rates_bp(theta, h, a, T):
    base, g = _split(theta, T)
    l1, l2 = P.linear_predictors(base, h, a, T)
    return np.exp(l1), np.exp(l2), float(np.exp(g))


def log_pmf(x, y, lam1, lam2, lam3):
    """Exact log P(X=x, Y=y) for arrays of scores (no truncation)."""
    D = _Data(np.asarray(x), np.asarray(y))
    return _terms(np.asarray(lam1, float), np.asarray(lam2, float), lam3, D)[0]


def score_grid_bp(lam1, lam2, lam3, max_goals=10):
    """Joint pmf grid (n, G, G), G = max_goals + 1, by convolving X1, Y1 with the shared Z."""
    lam1, lam2 = np.atleast_1d(lam1).astype(float), np.atleast_1d(lam2).astype(float)
    g = np.arange(max_goals + 1)
    p1 = np.exp(-lam1[:, None] + g * np.log(lam1[:, None]) - gammaln(g + 1))
    p2 = np.exp(-lam2[:, None] + g * np.log(lam2[:, None]) - gammaln(g + 1))
    p3 = np.exp(-lam3 + g * np.log(lam3) - gammaln(g + 1))
    out = np.zeros((len(lam1), max_goals + 1, max_goals + 1))
    for z in range(max_goals + 1):
        out[:, z:, z:] += p3[z] * p1[:, : max_goals + 1 - z, None] * p2[:, None, : max_goals + 1 - z]
    return out


def outcome_probs_from_grid(grid, normalise=True):
    n, G, _ = grid.shape
    i, j = np.indices((G, G))
    pH = (grid * (i > j)).sum(axis=(1, 2))
    pD = (grid * (i == j)).sum(axis=(1, 2))
    pA = (grid * (i < j)).sum(axis=(1, 2))
    out = np.column_stack([pH, pD, pA])
    return out / out.sum(axis=1, keepdims=True) if normalise else out


def outcome_probs_bp(lam1, lam2, lam3, max_goals=10):
    return outcome_probs_from_grid(score_grid_bp(lam1, lam2, lam3, max_goals))


def fit_before_bp(df, ref_date, T, xi, kappa, theta0=None):
    ref = pd.Timestamp(ref_date)
    train = df[df["Date"] < ref]
    if len(train) == 0:
        raise ValueError(f"No matches before {ref.date()}")
    w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi)
    return fit_bp(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, T, kappa, theta0=theta0)


def walk_forward_pair(df, T, start, end, xi, kappa):
    """Weekly walk-forward for Model B and the bivariate model on the same matches and the same data.

    Returns a DataFrame indexed like df with, for each model m in {b, bp}: pH_m, pD_m, pA_m, the exact log
    probability of the scoreline that happened (ls_m), and the likeliest scoreline (mode_x_m, mode_y_m);
    plus lam3 (fitted shared-tempo rate) per block.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    th_b = th_bp = None
    rows = []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = df[df["Date"] < ref]
        w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi)
        rb = P.fit(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, T, kappa, theta0=th_b)
        th_b = rb.theta
        rp = fit_bp(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, T, kappa, theta0=th_bp)
        th_bp = rp.theta
        h, a = block["h"].values, block["a"].values
        x, y = block["FTHG"].values, block["FTAG"].values
        lam, mu = P.rates(th_b, h, a, T)
        pb = P.outcome_probs(lam, mu)
        gb = P._pmf_grid(lam, mu, 10) if hasattr(P, "_pmf_grid") else None
        l1, l2, l3 = rates_bp(th_bp, h, a, T)
        gp = score_grid_bp(l1, l2, l3)
        pp = outcome_probs_from_grid(gp)
        ls_b = P_logpmf_indep(x, y, lam, mu)
        ls_p = log_pmf(x, y, l1, l2, l3)
        for i, idx in enumerate(block.index):
            mb = np.unravel_index(np.argmax(gb[i]), gb[i].shape)
            mp = np.unravel_index(np.argmax(gp[i]), gp[i].shape)
            rows.append({"idx": idx, "pH_b": pb[i, 0], "pD_b": pb[i, 1], "pA_b": pb[i, 2], "ls_b": ls_b[i],
                         "mode_x_b": mb[0], "mode_y_b": mb[1],
                         "pH_bp": pp[i, 0], "pD_bp": pp[i, 1], "pA_bp": pp[i, 2], "ls_bp": ls_p[i],
                         "mode_x_bp": mp[0], "mode_y_bp": mp[1], "lam3": l3})
    return pd.DataFrame(rows).set_index("idx").sort_index()


def P_logpmf_indep(x, y, lam, mu):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return x * np.log(lam) - lam - gammaln(x + 1) + y * np.log(mu) - mu - gammaln(y + 1)


def summarise(df, wf, name):
    """Paired comparison of bivariate vs Model B on one window."""
    y = df.loc[wf.index, "y"].values
    xg, yg = df.loc[wf.index, "FTHG"].values, df.loc[wf.index, "FTAG"].values
    pb = wf[["pH_b", "pD_b", "pA_b"]].values
    pp = wf[["pH_bp", "pD_bp", "pA_bp"]].values
    llb = -np.log(pb[np.arange(len(y)), y])
    llp = -np.log(pp[np.arange(len(y)), y])
    d = llp - llb
    dls = wf["ls_bp"].values - wf["ls_b"].values
    n = len(y)
    return {
        "window": name, "n": n,
        "LL_B": llb.mean(), "LL_BP": llp.mean(), "dLL": d.mean(), "dLL_se": d.std(ddof=1) / np.sqrt(n),
        "Brier_B": M.brier_score(pb, y), "Brier_BP": M.brier_score(pp, y),
        "score_LL_B": wf["ls_b"].mean(), "score_LL_BP": wf["ls_bp"].mean(), "dScore": dls.mean(), "dScore_se": dls.std(ddof=1) / np.sqrt(n),
        "draw_pred_B": pb[:, 1].mean(), "draw_pred_BP": pp[:, 1].mean(), "draw_act": float((y == 1).mean()),
        "exact_B": float(((wf["mode_x_b"].values == xg) & (wf["mode_y_b"].values == yg)).mean()),
        "exact_BP": float(((wf["mode_x_bp"].values == xg) & (wf["mode_y_bp"].values == yg)).mean()),
        "mode11_B": float(((wf["mode_x_b"] == 1) & (wf["mode_y_b"] == 1)).mean()),
        "mode11_BP": float(((wf["mode_x_bp"] == 1) & (wf["mode_y_bp"] == 1)).mean()),
        "lam3_mean": float(wf["lam3"].mean()),
    }


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    import time
    from .walkforward import load_matches
    ap = argparse.ArgumentParser(description="Bivariate Poisson vs independent Poisson, walk-forward.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--windows", default="validation,early,test")
    ap.add_argument("--out", default="reports/bivariate")
    args = ap.parse_args(argv)
    df, teams = load_matches(args.data)
    T = len(teams)
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.windows.split(","):
        s, e = WINDOWS[name]
        t0 = time.time()
        wf = walk_forward_pair(df, T, s, e, args.xi, args.kappa)
        r = summarise(df, wf, name)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        wf.to_csv(out / f"walkforward_{name}.csv")
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()

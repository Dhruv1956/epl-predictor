"""Independent-Poisson model for football match goals (Model B) and its extensions (Model C).

Model B
-------
For match i with home team h(i) and away team a(i), goals are independent Poisson:

    X_i ~ Pois(lam_i),   log lam_i = iota + eta + alpha[h(i)] - beta[a(i)]    (home goals)
    Y_i ~ Pois(mu_i),    log mu_i  = iota + alpha[a(i)] - beta[h(i)]          (away goals)

    iota  league-wide log goal rate        eta   home advantage
    alpha attack strength of each team     beta  defensive strength (larger = concedes less)

Objective: the weighted, ridge-penalised Poisson log-likelihood

    ell(theta) = sum_i w_i [ x_i log lam_i - lam_i + y_i log mu_i - mu_i - log(x_i! y_i!) ]
                 - kappa * ( sum_t alpha_t^2 + sum_t beta_t^2 )

Write e_i = x_i - lam_i and f_i = y_i - mu_i. The gradient, derived by hand (no autodiff), is

    d ell / d iota    =  sum_i w_i (e_i + f_i)
    d ell / d eta     =  sum_i w_i e_i
    d ell / d alpha_t =  sum_{i: h(i)=t} w_i e_i + sum_{i: a(i)=t} w_i f_i - 2 kappa alpha_t
    d ell / d beta_t  = -sum_{i: a(i)=t} w_i e_i - sum_{i: h(i)=t} w_i f_i - 2 kappa beta_t

Weights w_i = exp(-xi * age_i) give recent matches more influence (time decay).

Model C extensions (both optional and off by default, so Model B is unchanged)
------------------------------------------------------------------------------
1. Covariates on the home advantage: log lam_i gains + X_i . gamma, where X holds one row per
   match (for example a 0/1 flag for matches played behind closed doors). The gradient gains
   d ell / d gamma_j = sum_i w_i e'_i X_ij.

2. Dixon-Coles low-score correction. The joint pmf is multiplied by tau(x, y; lam, mu, rho):

       tau = 1 - lam mu rho   at (0, 0)        tau = 1 + lam rho   at (0, 1)
       tau = 1 + mu rho       at (1, 0)        tau = 1 - rho       at (1, 1)
       tau = 1 otherwise

   and ell gains sum_i w_i log tau_i. Because log lam and log mu enter only through the
   Poisson terms and tau, the gradient formulas above still hold with the residuals replaced by

       e'_i = e_i + d log tau_i / d log lam_i,     f'_i = f_i + d log tau_i / d log mu_i

   where, with tau evaluated at the match's own score,

       (0,0): d log tau / d log lam = d log tau / d log mu = -lam mu rho / tau
       (0,1): d log tau / d log lam = lam rho / tau          (1,0): d log tau / d log mu = mu rho / tau
       all other derivatives with respect to log lam, log mu are 0

   and d log tau / d rho = -lam mu / tau, lam / tau, mu / tau, -1 / tau at (0,0), (0,1), (1,0), (1,1).
   Unlike Model B, the likelihood with rho != 0 is not concave, so the maximiser is not
   guaranteed unique; rho is kept in a box [-0.2, 0.2] that keeps every tau positive in practice.

Parameter vector layout:  theta = [iota, eta, alpha_0..alpha_{T-1}, beta_0..beta_{T-1}, gamma_1..gamma_k, rho]
where the gamma block is present only if X has k columns and rho only if dc=True.
Only NumPy and SciPy are used; scipy.optimize supplies the L-BFGS-B optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

RHO_BOUNDS = (-0.2, 0.2)
_TAU_FLOOR = 1e-10  # keeps log(tau) finite at invalid trial points the line search will reject

# ----------------------------------------------------------------------
# Parameters and rates
# ----------------------------------------------------------------------


def n_params(T, k=0, dc=False):
    """Length of theta: 2 + 2T base parameters, k gammas, and rho if dc."""
    return 2 + 2 * T + k + (1 if dc else 0)


def unpack(theta, T):
    """Split theta into (iota, eta, alpha, beta). Extra parameters, if any, are ignored."""
    return theta[0], theta[1], theta[2 : 2 + T], theta[2 + T : 2 + 2 * T]


def unpack_extra(theta, T, k=0, dc=False):
    """Return (gamma, rho): the covariate coefficients and the Dixon-Coles parameter (0 if unused)."""
    gamma = theta[2 + 2 * T : 2 + 2 * T + k]
    rho = float(theta[2 + 2 * T + k]) if dc else 0.0
    return gamma, rho


def linear_predictors(theta, h, a, T, X=None, Xm=None):
    """Return (log lam, log mu) for each match.

    X (n, k) adds X @ gamma to log lam. Xm (n, k), if given, adds Xm @ gamma to log mu with the
    SAME gamma (so a coefficient can act on both teams' rates, e.g. a newly-promoted-team effect).
    """
    iota, eta, alpha, beta = unpack(theta, T)
    log_lam = iota + eta + alpha[h] - beta[a]
    log_mu = iota + alpha[a] - beta[h]
    if X is not None and X.shape[1] > 0:
        gamma = theta[2 + 2 * T : 2 + 2 * T + X.shape[1]]
        log_lam = log_lam + X @ gamma
        if Xm is not None:
            log_mu = log_mu + Xm @ gamma
    elif Xm is not None:
        raise ValueError("Xm requires X")
    return log_lam, log_mu


def rates(theta, h, a, T, X=None, Xm=None):
    """Return the Poisson means (lam, mu) for each match."""
    log_lam, log_mu = linear_predictors(theta, h, a, T, X, Xm)
    return np.exp(log_lam), np.exp(log_mu)


# ----------------------------------------------------------------------
# Time decay
# ----------------------------------------------------------------------


def age_in_years(dates, ref_date):
    """Age of each match in years at ref_date. Dates are numpy datetime64."""
    days = (np.datetime64(ref_date, "D") - np.asarray(dates, dtype="datetime64[D]")).astype(float)
    return days / 365.25


def time_weights(age_years, xi):
    """Exponential time-decay weights w = exp(-xi * age). Half-life is ln(2)/xi years."""
    return np.exp(-xi * np.asarray(age_years, dtype=float))


# ----------------------------------------------------------------------
# Dixon-Coles correction
# ----------------------------------------------------------------------


def dc_tau_terms(x, y, lam, mu, rho):
    """tau and its derivatives for each match, evaluated at the match's own score (x, y).

    Returns (tau, d_loglam, d_logmu, d_rho) where d_loglam = d log tau / d log lam and so on.
    """
    tau = np.ones_like(lam, dtype=float)
    d_loglam = np.zeros_like(tau)
    d_logmu = np.zeros_like(tau)
    d_rho = np.zeros_like(tau)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    tau = np.maximum(tau, _TAU_FLOOR)

    d_loglam[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
    d_logmu[m00] = d_loglam[m00]
    d_loglam[m01] = lam[m01] * rho / tau[m01]
    d_logmu[m10] = mu[m10] * rho / tau[m10]

    d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
    d_rho[m01] = lam[m01] / tau[m01]
    d_rho[m10] = mu[m10] / tau[m10]
    d_rho[m11] = -1.0 / tau[m11]
    return tau, d_loglam, d_logmu, d_rho


def _apply_tau(joint, lam, mu, rho):
    """Multiply the four lowest scorelines of a joint pmf grid (n, x, y) by the Dixon-Coles tau."""
    if rho != 0.0:
        joint[:, 0, 0] *= 1.0 - lam * mu * rho
        joint[:, 0, 1] *= 1.0 + lam * rho
        joint[:, 1, 0] *= 1.0 + mu * rho
        joint[:, 1, 1] *= 1.0 - rho
    return joint


# ----------------------------------------------------------------------
# Objective and hand-derived gradient
# ----------------------------------------------------------------------


def loglik_and_grad(theta, h, a, x, y, w, T, kappa, X=None, dc=False, Xm=None):
    """Penalised weighted log-likelihood ell(theta) and its gradient.

    h, a : integer team indices (0..T-1) of home and away team per match
    x, y : home and away goals
    w    : non-negative match weights
    X    : optional (n, k) covariates entering log lam (with coefficients gamma)
    Xm   : optional (n, k) covariates entering log mu with the same gamma
    dc   : include the Dixon-Coles correction with rho as the last parameter
    """
    k = 0 if X is None else X.shape[1]
    log_lam, log_mu = linear_predictors(theta, h, a, T, X, Xm)
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    _, _, alpha, beta = unpack(theta, T)

    ll = np.sum(w * (x * log_lam - lam + y * log_mu - mu - gammaln(x + 1) - gammaln(y + 1)))
    ll -= kappa * (alpha @ alpha + beta @ beta)

    e = x - lam
    f = y - mu
    if dc:
        rho = theta[2 + 2 * T + k]
        tau, d_loglam, d_logmu, d_rho = dc_tau_terms(x, y, lam, mu, rho)
        ll += np.sum(w * np.log(tau))
        e = e + d_loglam
        f = f + d_logmu

    grad = np.zeros_like(theta, dtype=float)
    grad[0] = np.sum(w * (e + f))
    grad[1] = np.sum(w * e)
    grad[2 : 2 + T] = (
        np.bincount(h, weights=w * e, minlength=T)
        + np.bincount(a, weights=w * f, minlength=T)
        - 2.0 * kappa * alpha
    )
    grad[2 + T : 2 + 2 * T] = (
        -np.bincount(a, weights=w * e, minlength=T)
        - np.bincount(h, weights=w * f, minlength=T)
        - 2.0 * kappa * beta
    )
    if k:
        grad[2 + 2 * T : 2 + 2 * T + k] = X.T @ (w * e)
        if Xm is not None:
            grad[2 + 2 * T : 2 + 2 * T + k] += Xm.T @ (w * f)
    if dc:
        grad[2 + 2 * T + k] = np.sum(w * d_rho)
    return ll, grad


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


@dataclass
class FitResult:
    theta: np.ndarray
    loglik: float  # penalised weighted log-likelihood at the optimum
    success: bool
    n_iter: int
    message: str

    def unpack(self, T: int):
        return unpack(self.theta, T)

    def unpack_extra(self, T: int, k: int = 0, dc: bool = False):
        return unpack_extra(self.theta, T, k, dc)


def fit(h, a, x, y, w, T, kappa, theta0=None, max_iter=1000, X=None, dc=False, Xm=None):
    """Maximise the penalised weighted log-likelihood with L-BFGS-B.

    kappa must be positive: the ridge penalty is what makes the Model B maximiser unique
    (the likelihood alone cannot separate iota from the means of alpha and beta).
    For Model B the objective is concave, so the optimum found is the global one.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0 for a unique maximiser")
    h, a = np.asarray(h), np.asarray(a)
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
    if Xm is not None:
        Xm = np.asarray(Xm, dtype=float)
        if Xm.ndim == 1:
            Xm = Xm[:, None]
    k = 0 if X is None else X.shape[1]

    if theta0 is None:
        theta0 = np.zeros(n_params(T, k, dc))
        mean_goals = np.average(np.concatenate([x, y]), weights=np.concatenate([w, w]))
        theta0[0] = np.log(max(mean_goals, 1e-3))
    bounds = None
    if dc:
        bounds = [(None, None)] * (len(theta0) - 1) + [RHO_BOUNDS]

    def neg(theta):
        ll, g = loglik_and_grad(theta, h, a, x, y, w, T, kappa, X=X, dc=dc, Xm=Xm)
        return -ll, -g

    res = minimize(
        neg,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-13, "gtol": 1e-8},
    )
    return FitResult(res.x, -res.fun, bool(res.success), int(res.nit), str(res.message))


# ----------------------------------------------------------------------
# From goal rates to match outcomes
# ----------------------------------------------------------------------


def _pmf_grid(lam, mu, max_goals):
    g = np.arange(max_goals + 1)
    pmf_h = np.exp(g[None, :] * np.log(lam)[:, None] - lam[:, None] - gammaln(g + 1)[None, :])
    pmf_a = np.exp(g[None, :] * np.log(mu)[:, None] - mu[:, None] - gammaln(g + 1)[None, :])
    return pmf_h[:, :, None] * pmf_a[:, None, :]  # shape (n, x, y)


def outcome_probs(lam, mu, max_goals=10, normalise=True, rho=0.0):
    """P(home win), P(draw), P(away win) from Poisson means, optionally Dixon-Coles corrected.

    Sums the joint pmf over the score grid 0..max_goals per side. With normalise=True the small
    truncated tail is redistributed so each row sums to 1. The tau correction itself conserves
    total probability exactly. Returns an array of shape (n_matches, 3) with columns [H, D, A].
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    joint = _apply_tau(_pmf_grid(lam, mu, max_goals), lam, mu, rho)
    g = np.arange(max_goals + 1)
    xi, yi = g[:, None], g[None, :]
    p_home = (joint * (xi > yi)).sum(axis=(1, 2))
    p_draw = (joint * (xi == yi)).sum(axis=(1, 2))
    p_away = (joint * (xi < yi)).sum(axis=(1, 2))
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    if normalise:
        probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# ----------------------------------------------------------------------
# Synthetic data (for testing that the fit recovers known parameters)
# ----------------------------------------------------------------------


def random_theta(T, rng, iota=0.18, eta=0.22, sd=0.25):
    """Draw a plausible true parameter vector. alpha and beta are centred to mean 0."""
    alpha = rng.normal(0.0, sd, T)
    beta = rng.normal(0.0, sd, T)
    return np.concatenate([[iota, eta], alpha - alpha.mean(), beta - beta.mean()])


def round_robin_fixtures(T, n_seasons):
    """Home and away team indices for n_seasons double round robins (T(T-1) matches each)."""
    home, away = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    keep = home != away
    return np.tile(home[keep], n_seasons), np.tile(away[keep], n_seasons)


def simulate_matches(theta, T, n_seasons, rng):
    """Simulate independent-Poisson double round-robin seasons (Model B).

    Returns h, a, x, y. With T = 20 each season has 380 matches, like the Premier League.
    """
    h, a = round_robin_fixtures(T, n_seasons)
    lam, mu = rates(theta, h, a, T)
    return h, a, rng.poisson(lam), rng.poisson(mu)


def simulate_scores(lam, mu, rng, rho=0.0, max_goals=10, chunk=4000):
    """Sample scores (x, y) from the Dixon-Coles joint pmf with given rates (rho=0 is plain Poisson)."""
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = len(lam)
    xs, ys = np.empty(n, dtype=int), np.empty(n, dtype=int)
    width = max_goals + 1
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        joint = _apply_tau(_pmf_grid(lam[sl], mu[sl], max_goals), lam[sl], mu[sl], rho)
        flat = joint.reshape(joint.shape[0], -1)
        cum = np.cumsum(flat / flat.sum(axis=1, keepdims=True), axis=1)
        u = rng.random((flat.shape[0], 1))
        idx = np.minimum((u > cum).sum(axis=1), flat.shape[1] - 1)
        xs[sl], ys[sl] = idx // width, idx % width
    return xs, ys

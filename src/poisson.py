"""Independent-Poisson model for football match goals (Model B).

Model
-----
For match i with home team h(i) and away team a(i), goals are independent Poisson:

    X_i ~ Pois(lam_i),   log lam_i = iota + eta + alpha[h(i)] - beta[a(i)]    (home goals)
    Y_i ~ Pois(mu_i),    log mu_i  = iota + alpha[a(i)] - beta[h(i)]          (away goals)

    iota  league-wide log goal rate        eta   home advantage
    alpha attack strength of each team     beta  defensive strength (larger = concedes less)

Parameter vector layout: theta = [iota, eta, alpha_0..alpha_{T-1}, beta_0..beta_{T-1}].

Objective
---------
The weighted, ridge-penalised Poisson log-likelihood

    ell(theta) = sum_i w_i [ x_i log lam_i - lam_i + y_i log mu_i - mu_i - log(x_i! y_i!) ]
                 - kappa * ( sum_t alpha_t^2 + sum_t beta_t^2 )

with residuals e_i = x_i - lam_i and f_i = y_i - mu_i, whose gradient (derived by hand,
no automatic differentiation) is

    d ell / d iota    =  sum_i w_i (e_i + f_i)
    d ell / d eta     =  sum_i w_i e_i
    d ell / d alpha_t =  sum_{i: h(i)=t} w_i e_i + sum_{i: a(i)=t} w_i f_i - 2 kappa alpha_t
    d ell / d beta_t  = -sum_{i: a(i)=t} w_i e_i - sum_{i: h(i)=t} w_i f_i - 2 kappa beta_t

Weights w_i = exp(-xi * age_i) give recent matches more influence (time decay).
Only NumPy and SciPy are used; scipy.optimize supplies the L-BFGS optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# ----------------------------------------------------------------------
# Parameters and rates
# ----------------------------------------------------------------------


def unpack(theta: np.ndarray, T: int):
    """Split theta into (iota, eta, alpha, beta)."""
    return theta[0], theta[1], theta[2 : 2 + T], theta[2 + T : 2 + 2 * T]


def linear_predictors(theta, h, a, T):
    """Return (log lam, log mu) for each match."""
    iota, eta, alpha, beta = unpack(theta, T)
    log_lam = iota + eta + alpha[h] - beta[a]
    log_mu = iota + alpha[a] - beta[h]
    return log_lam, log_mu


def rates(theta, h, a, T):
    """Return the Poisson means (lam, mu) for each match."""
    log_lam, log_mu = linear_predictors(theta, h, a, T)
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
# Objective and hand-derived gradient
# ----------------------------------------------------------------------


def loglik_and_grad(theta, h, a, x, y, w, T, kappa):
    """Penalised weighted log-likelihood ell(theta) and its gradient.

    h, a : integer team indices (0..T-1) of home and away team per match
    x, y : home and away goals
    w    : non-negative match weights
    """
    log_lam, log_mu = linear_predictors(theta, h, a, T)
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    _, _, alpha, beta = unpack(theta, T)

    ll = np.sum(w * (x * log_lam - lam + y * log_mu - mu - gammaln(x + 1) - gammaln(y + 1)))
    ll -= kappa * (alpha @ alpha + beta @ beta)

    e = x - lam
    f = y - mu
    grad = np.empty_like(theta, dtype=float)
    grad[0] = np.sum(w * (e + f))
    grad[1] = np.sum(w * e)
    grad[2 : 2 + T] = (
        np.bincount(h, weights=w * e, minlength=T)
        + np.bincount(a, weights=w * f, minlength=T)
        - 2.0 * kappa * alpha
    )
    grad[2 + T :] = (
        -np.bincount(a, weights=w * e, minlength=T)
        - np.bincount(h, weights=w * f, minlength=T)
        - 2.0 * kappa * beta
    )
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


def fit(h, a, x, y, w, T, kappa, theta0=None, max_iter=1000):
    """Maximise the penalised weighted log-likelihood with L-BFGS.

    kappa must be positive: the ridge penalty is what makes the maximiser unique
    (the likelihood alone cannot separate iota from the means of alpha and beta).
    The objective is concave, so the optimum found is the global one.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0 for a unique maximiser")
    h, a = np.asarray(h), np.asarray(a)
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))

    if theta0 is None:
        theta0 = np.zeros(2 + 2 * T)
        mean_goals = np.average(np.concatenate([x, y]), weights=np.concatenate([w, w]))
        theta0[0] = np.log(max(mean_goals, 1e-3))

    def neg(theta):
        ll, g = loglik_and_grad(theta, h, a, x, y, w, T, kappa)
        return -ll, -g

    res = minimize(
        neg,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": 10 * max_iter, "ftol": 1e-13, "gtol": 1e-8},
    )
    return FitResult(res.x, -res.fun, bool(res.success), int(res.nit), str(res.message))


# ----------------------------------------------------------------------
# From goal rates to match outcomes
# ----------------------------------------------------------------------


def outcome_probs(lam, mu, max_goals=10, normalise=True):
    """P(home win), P(draw), P(away win) from independent Poisson means.

    Sums the joint pmf P(x, y) = Pois(x; lam) Pois(y; mu) over the score grid
    0..max_goals per side. With normalise=True the small truncated tail is
    redistributed so each row sums to 1.
    Returns an array of shape (n_matches, 3) with columns [H, D, A].
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    g = np.arange(max_goals + 1)
    pmf_h = np.exp(g[None, :] * np.log(lam)[:, None] - lam[:, None] - gammaln(g + 1)[None, :])
    pmf_a = np.exp(g[None, :] * np.log(mu)[:, None] - mu[:, None] - gammaln(g + 1)[None, :])
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]  # shape (n, x, y)
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


def simulate_matches(theta, T, n_seasons, rng):
    """Simulate double round-robin seasons (every ordered pair plays once per season).

    Returns h, a, x, y. With T = 20 each season has 380 matches, like the Premier League.
    """
    home, away = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    keep = home != away
    h_one, a_one = home[keep], away[keep]
    h = np.tile(h_one, n_seasons)
    a = np.tile(a_one, n_seasons)
    lam, mu = rates(theta, h, a, T)
    x = rng.poisson(lam)
    y = rng.poisson(mu)
    return h, a, x, y

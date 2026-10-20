"""Tests for src/poisson.py: the independent-Poisson model (Model B).

Every test uses simulated data with known parameters, so the right answer is known.
"""

import numpy as np
import pytest
from scipy.stats import poisson, skellam

from src import poisson as P

T = 20  # teams, as in the Premier League


def make_data(seed, n_seasons):
    rng = np.random.default_rng(seed)
    theta_true = P.random_theta(T, rng)
    h, a, x, y = P.simulate_matches(theta_true, T, n_seasons, rng)
    return rng, theta_true, h, a, x, y


# ----------------------------------------------------------------------
# Likelihood and gradient
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_loglik_value_matches_scipy_poisson(kappa):
    rng, theta_true, h, a, x, y = make_data(0, 2)
    w = rng.uniform(0.1, 1.0, len(h))
    theta = theta_true + rng.normal(0, 0.1, theta_true.size)
    ll, _ = P.loglik_and_grad(theta, h, a, x, y, w, T, kappa)
    lam, mu = P.rates(theta, h, a, T)
    expected = np.sum(w * (poisson.logpmf(x, lam) + poisson.logpmf(y, mu)))
    expected -= kappa * np.sum(theta[2:] ** 2)
    assert ll == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_gradient_matches_finite_differences(kappa):
    rng, theta_true, h, a, x, y = make_data(1, 2)
    w = rng.uniform(0.1, 1.0, len(h))  # arbitrary non-uniform weights
    theta = theta_true + rng.normal(0, 0.1, theta_true.size)  # a point away from the optimum
    _, grad = P.loglik_and_grad(theta, h, a, x, y, w, T, kappa)

    eps = 1e-6
    numeric = np.empty_like(grad)
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = eps
        up = P.loglik_and_grad(theta + step, h, a, x, y, w, T, kappa)[0]
        down = P.loglik_and_grad(theta - step, h, a, x, y, w, T, kappa)[0]
        numeric[k] = (up - down) / (2 * eps)

    scale = max(1.0, np.max(np.abs(numeric)))
    assert np.max(np.abs(grad - numeric)) / scale < 1e-6


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def test_fit_reaches_stationary_point_and_ridge_pins_means_to_zero():
    _, theta_true, h, a, x, y = make_data(3, 5)
    w = np.ones(len(h))
    kappa = 0.5
    result = P.fit(h, a, x, y, w, T, kappa)
    assert result.success

    _, grad = P.loglik_and_grad(result.theta, h, a, x, y, w, T, kappa)
    assert np.max(np.abs(grad)) < 1e-4

    # At the optimum both strength vectors have mean zero (see the identity test below).
    # L-BFGS stops at floating-point precision, so we allow 1e-4 rather than exact zero.
    _, _, alpha, beta = result.unpack(T)
    assert abs(alpha.sum()) < 1e-4
    assert abs(beta.sum()) < 1e-4


def test_gradient_sums_identity_that_pins_the_strength_means():
    # Every match has one home and one away team, so summing the gradients over teams gives
    #   sum_t d ell/d alpha_t =  d ell/d iota - 2 kappa sum_t alpha_t
    #   sum_t d ell/d beta_t  = -d ell/d iota - 2 kappa sum_t beta_t
    # at ANY theta. Setting the gradient to zero then forces sum alpha = sum beta = 0.
    rng, theta_true, h, a, x, y = make_data(8, 2)
    w = rng.uniform(0.1, 1.0, len(h))
    theta = theta_true + rng.normal(0, 0.2, theta_true.size)
    kappa = 0.7
    _, g = P.loglik_and_grad(theta, h, a, x, y, w, T, kappa)
    _, _, alpha, beta = P.unpack(theta, T)
    assert g[2 : 2 + T].sum() == pytest.approx(g[0] - 2 * kappa * alpha.sum(), abs=1e-8)
    assert g[2 + T :].sum() == pytest.approx(-g[0] - 2 * kappa * beta.sum(), abs=1e-8)


def test_fit_beats_true_parameters_on_the_objective():
    _, theta_true, h, a, x, y = make_data(4, 3)
    w = np.ones(len(h))
    kappa = 0.5
    result = P.fit(h, a, x, y, w, T, kappa)
    ll_true, _ = P.loglik_and_grad(theta_true, h, a, x, y, w, T, kappa)
    assert result.loglik >= ll_true - 1e-9


def test_zero_weight_matches_are_ignored():
    _, _, h, a, x, y = make_data(5, 3)
    keep = np.arange(len(h)) % 2 == 0
    full = P.fit(h, a, x, y, keep.astype(float), T, 0.5)
    subset = P.fit(h[keep], a[keep], x[keep], y[keep], np.ones(keep.sum()), T, 0.5)
    assert np.allclose(full.theta, subset.theta, atol=1e-4)


def test_fit_requires_positive_kappa():
    _, _, h, a, x, y = make_data(6, 1)
    with pytest.raises(ValueError):
        P.fit(h, a, x, y, np.ones(len(h)), T, 0.0)


def strength_rmse(n_seasons, seeds):
    errors = []
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        theta_true = P.random_theta(T, rng)
        h, a, x, y = P.simulate_matches(theta_true, T, n_seasons, rng)
        result = P.fit(h, a, x, y, np.ones(len(h)), T, kappa=0.5)
        errors.append(np.sqrt(np.mean((result.theta[2:] - theta_true[2:]) ** 2)))
    return float(np.mean(errors))


def test_parameter_recovery_improves_with_more_data():
    # Standard errors shrink like 1/sqrt(n): ten seasons should be about 3x more accurate
    # than one. We only demand a clear improvement and a small absolute error.
    one_season = strength_rmse(1, range(5))
    ten_seasons = strength_rmse(10, range(5))
    assert ten_seasons < 0.6 * one_season
    assert ten_seasons < 0.08


def test_league_level_and_home_advantage_are_recovered():
    _, theta_true, h, a, x, y = make_data(7, 10)
    result = P.fit(h, a, x, y, np.ones(len(h)), T, kappa=0.5)
    iota_hat, eta_hat, _, _ = result.unpack(T)
    assert abs(eta_hat - theta_true[1]) < 0.06
    assert abs(iota_hat - theta_true[0]) < 0.1


# ----------------------------------------------------------------------
# Probabilities and weights
# ----------------------------------------------------------------------


def test_outcome_probs_match_exact_skellam_distribution():
    lam = np.array([0.6, 1.4, 2.8, 1.0])
    mu = np.array([1.9, 1.1, 0.7, 1.0])
    probs = P.outcome_probs(lam, mu, max_goals=40, normalise=False)
    exact = np.stack([skellam.sf(0, lam, mu), skellam.pmf(0, lam, mu), skellam.cdf(-1, lam, mu)], axis=1)
    assert np.allclose(probs, exact, atol=1e-9)


def test_outcome_probs_sum_to_one_and_respond_to_strength():
    probs = P.outcome_probs(np.array([1.5, 0.8]), np.array([1.0, 1.8]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > probs[0, 2]  # stronger home side: home win more likely than away win
    assert probs[1, 2] > probs[1, 0]  # stronger away side: the reverse


def test_time_weights_and_ages():
    xi = 0.5
    assert P.time_weights(np.log(2) / xi, xi) == pytest.approx(0.5)  # half-life ln2 / xi
    ages = P.age_in_years(np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[D]"), "2022-01-01")
    assert np.allclose(ages, [2.0, 1.0], atol=0.01)

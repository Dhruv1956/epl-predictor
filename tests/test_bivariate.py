"""Bivariate Poisson: gradient, limits and a recovery check on simulated data."""
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import poisson

from src import bivariate as BP
from src import poisson as P


def _toy(T=5, n=300, lam3=0.25, seed=3):
    rng = np.random.default_rng(seed)
    theta = np.concatenate([P.random_theta(T, rng), [np.log(lam3)]])
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    l1, l2, l3 = BP.rates_bp(theta, h, a, T)
    z = rng.poisson(l3, n)
    x, y = rng.poisson(l1) + z, rng.poisson(l2) + z
    return theta, h, a, x, y, T


def test_gradient_matches_finite_differences():
    theta, h, a, x, y, T = _toy()
    w = np.random.default_rng(1).uniform(0.3, 1.0, len(x))
    rng = np.random.default_rng(2)
    t0 = theta + rng.normal(0, 0.05, len(theta))
    D = BP._Data(x, y)
    _, g = BP.loglik_and_grad_bp(t0, h, a, x, y, w, T, 5.0, D)
    num = approx_fprime(t0, lambda t: BP.loglik_and_grad_bp(t, h, a, x, y, w, T, 5.0, D)[0], 1e-6)
    assert np.max(np.abs(g - num)) < 1e-4 * max(1, np.max(np.abs(g)))


def test_reduces_to_independent_poisson_when_lam3_tiny():
    theta, h, a, x, y, T = _toy()
    t = theta.copy()
    t[-1] = np.log(1e-9)
    w = np.ones(len(x))
    ll_bp, _ = BP.loglik_and_grad_bp(t, h, a, x, y, w, T, 5.0)
    ll_b, _ = P.loglik_and_grad(t[:-1], h, a, x, y, w, T, 5.0)
    assert ll_bp == pytest.approx(ll_b, abs=1e-5)


def test_pmf_is_a_distribution_with_the_right_moments():
    l1, l2, l3 = np.array([1.3]), np.array([0.9]), 0.3
    g = BP.score_grid_bp(l1, l2, l3, max_goals=25)[0]
    i = np.arange(26)
    assert g.sum() == pytest.approx(1.0, abs=1e-10)
    assert (g.sum(axis=1) @ i) == pytest.approx(l1[0] + l3, abs=1e-8)  # E[X]
    assert (g.sum(axis=0) @ i) == pytest.approx(l2[0] + l3, abs=1e-8)  # E[Y]
    cov = (g * np.outer(i, i)).sum() - (g.sum(axis=1) @ i) * (g.sum(axis=0) @ i)
    assert cov == pytest.approx(l3, abs=1e-8)  # Cov(X, Y) = lam3


def test_closed_form_pmf_agrees_with_the_convolution_grid():
    l1, l2, l3 = np.array([1.1, 2.0]), np.array([0.8, 1.4]), 0.2
    grid = BP.score_grid_bp(l1, l2, l3, max_goals=10)
    for xs, ys in [(0, 0), (1, 1), (2, 1), (3, 4)]:
        lp = BP.log_pmf(np.array([xs, xs]), np.array([ys, ys]), l1, l2, l3)
        assert np.exp(lp) == pytest.approx(grid[:, xs, ys], rel=1e-10)


def test_outcome_probs_rows_sum_to_one_and_lam3_raises_draws():
    l1, l2 = np.array([1.5, 1.0]), np.array([1.1, 1.0])
    p0 = BP.outcome_probs_bp(l1, l2, 1e-6)
    p1 = BP.outcome_probs_bp(l1 - 0.3, l2 - 0.3, 0.3)  # same marginal means, positive correlation
    assert np.allclose(p0.sum(axis=1), 1) and np.allclose(p1.sum(axis=1), 1)
    ref = P.outcome_probs(l1, l2)
    assert np.allclose(p0, ref, atol=1e-5)
    assert np.all(p1[:, 1] > ref[:, 1])


def test_fit_recovers_the_shared_component():
    theta, h, a, x, y, T = _toy(T=6, n=20000, lam3=0.3, seed=11)
    r = BP.fit_bp(h, a, x, y, np.ones(len(x)), T, kappa=0.5)
    assert r.success
    assert r.lam3 == pytest.approx(0.3, abs=0.04)

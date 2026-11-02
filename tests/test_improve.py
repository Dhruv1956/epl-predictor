"""Calibration and split-kappa pieces."""
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from src import improve as I
from src import poisson as P


def _forecasts(n=800, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.6, (n, 3))
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in p])
    return p, y


def test_identity_params_leave_probabilities_unchanged():
    p, _ = _forecasts()
    assert np.allclose(I.recalibrate(p, (1.0, 0.0, 0.0)), p, atol=1e-9)
    assert np.allclose(I.recalibrate(p, (0.7, 0.2, -0.1)).sum(axis=1), 1)


def test_recalibration_gradient_matches_finite_differences():
    p, y = _forecasts()
    th = np.array([0.9, 0.15, -0.1])
    _, g = I.recal_loglik_and_grad(th, p, y)
    num = approx_fprime(th, lambda t: I.recal_loglik_and_grad(t, p, y)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)


def test_recalibration_recovers_a_known_distortion():
    rng = np.random.default_rng(5)
    n = 20000
    z = rng.normal(0, 0.7, (n, 3))
    true_p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in true_p])
    overconfident = I.recalibrate(true_p, (1.6, 0.0, 0.0))  # forecasts that are too sharp
    t, bH, bD = I.fit_recalibration(overconfident, y)
    assert t == pytest.approx(1 / 1.6, abs=0.06)
    assert abs(bH) < 0.1 and abs(bD) < 0.1


def test_split_kappa_equals_model_b_when_kappas_are_equal():
    rng = np.random.default_rng(2)
    T, n = 6, 400
    theta = P.random_theta(T, rng)
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    lam, mu = P.rates(theta, h, a, T)
    x, y = rng.poisson(lam), rng.poisson(mu)
    w = np.ones(n)
    ref = P.fit(h, a, x, y, w, T, 5.0).theta
    got = I.fit_split(h, a, x, y, w, T, 5.0, 5.0)
    assert np.allclose(ref, got, atol=1e-4)


def test_split_kappa_shrinks_the_penalised_side_more():
    rng = np.random.default_rng(3)
    T, n = 6, 400
    theta = P.random_theta(T, rng)
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    lam, mu = P.rates(theta, h, a, T)
    x, y = rng.poisson(lam), rng.poisson(mu)
    w = np.ones(n)
    th = I.fit_split(h, a, x, y, w, T, 1.0, 100.0)
    _, _, alpha, beta = P.unpack(th, T)
    assert np.sum(beta ** 2) < np.sum(alpha ** 2)

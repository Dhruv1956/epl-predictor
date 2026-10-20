"""Tests for per-team home advantage delta_t."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src import poisson as P
from src import walkforward as W
from tests.test_walkforward import synthetic_league


def _setup(T=6, seed=0, with_x=True, dc=True):
    rng = np.random.default_rng(seed)
    h, a = P.round_robin_fixtures(T, 5)
    n = len(h)
    k = 2 if with_x else 0
    X = Xm = None
    if with_x:
        hn, an = (rng.random(n) < 0.2).astype(float), (rng.random(n) < 0.2).astype(float)
        X, Xm = np.column_stack([hn, -an]), np.column_stack([an, -hn])
    theta = np.concatenate([P.random_theta(T, rng, sd=0.3), rng.normal(-0.1, 0.05, k), rng.normal(0, 0.1, T), [-0.08] if dc else []])
    lam, mu = P.rates(theta, h, a, T, X, Xm, ha=True)
    x, y = rng.poisson(lam), rng.poisson(mu)
    return theta, h, a, x, y, X, Xm, T


@pytest.mark.parametrize("with_x,dc", [(False, False), (True, False), (True, True)])
def test_gradient_with_team_home_advantage(with_x, dc):
    theta, h, a, x, y, X, Xm, T = _setup(with_x=with_x, dc=dc)
    w = np.random.default_rng(1).uniform(0.3, 1.0, len(h))
    t0 = theta + np.random.default_rng(2).normal(0, 0.03, theta.size)
    f = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, dc=dc, Xm=Xm, kappa_h=15.0)[0]
    g = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, dc=dc, Xm=Xm, kappa_h=15.0)[1]
    assert check_grad(f, g, t0, epsilon=1e-6) < 1e-4 * max(1.0, np.linalg.norm(g(t0)))


def test_huge_ridge_collapses_to_common_home_advantage():
    theta, h, a, x, y, X, Xm, T = _setup(with_x=False, dc=False)
    w = np.ones(len(h))
    plain = P.fit(h, a, x, y, w, T, 2.0)
    shrunk = P.fit(h, a, x, y, w, T, 2.0, kappa_h=1e4)
    np.testing.assert_allclose(shrunk.unpack_delta(T, 0), 0.0, atol=2e-3)
    np.testing.assert_allclose(shrunk.theta[: 2 + 2 * T], plain.theta, atol=5e-3)


def test_recovers_team_home_advantages():
    T = 8
    rng = np.random.default_rng(3)
    h, a = P.round_robin_fixtures(T, 120)
    delta = rng.normal(0, 0.15, T)
    delta -= delta.mean()
    theta = np.concatenate([P.random_theta(T, rng, sd=0.25), delta])
    lam, mu = P.rates(theta, h, a, T, ha=True)
    x, y = rng.poisson(lam), rng.poisson(mu)
    res = P.fit(h, a, x, y, np.ones(len(h)), T, 0.5, kappa_h=0.5)
    est = res.unpack_delta(T, 0)
    assert np.corrcoef(est, delta)[0, 1] > 0.95
    assert np.max(np.abs((est - est.mean()) - delta)) < 0.1  # SE is about 0.03 per team


def test_no_leakage_with_team_home_advantage():
    df, _ = synthetic_league(n_seasons=5, seed=9)
    df["closed_doors"] = 0
    df["h_new"] = 0
    df["a_new"] = 0
    spec = W.ModelSpec("ha", kappa_home=20.0)
    start, end = "2019-08-01", "2020-08-01"
    p1 = W.walk_forward(df, 10, start, end, 0.5, 2.0, spec=spec)
    df2 = df.copy()
    cut = pd.Timestamp("2020-02-01")
    late = df2["Date"] >= cut
    df2.loc[late, ["FTHG", "FTAG"]] = df2.loc[late, ["FTAG", "FTHG"]].values + 3
    df2["y"] = np.where(df2.FTHG > df2.FTAG, 0, np.where(df2.FTHG == df2.FTAG, 1, 2))
    p2 = W.walk_forward(df2, 10, start, end, 0.5, 2.0, spec=spec)
    idx = p1.index[(df["Date"] < cut).loc[p1.index].values]
    np.testing.assert_allclose(p1.loc[idx].values, p2.loc[idx].values, atol=1e-7)

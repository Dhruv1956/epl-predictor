"""Tests for Model C: covariates (closed doors) and the Dixon-Coles correction."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src import poisson as P
from src import walkforward as W
from tests.test_walkforward import synthetic_league

T = 6


def _data(n=300, seed=0, rho=-0.1, gamma=-0.15):
    rng = np.random.default_rng(seed)
    base = P.random_theta(T, rng, sd=0.3)
    h, a = P.round_robin_fixtures(T, 1 + n // (T * (T - 1)))
    h, a = h[:n], a[:n]
    X = (rng.random((len(h), 1)) < 0.4).astype(float)
    theta = np.concatenate([base, [gamma, rho]])
    lam, mu = P.rates(theta, h, a, T, X)
    x, y = P.simulate_scores(lam, mu, rng, rho=rho)
    return theta, h, a, x, y, X


def test_gradient_matches_finite_differences_with_covariate_and_dc():
    theta, h, a, x, y, X = _data()
    w = np.random.default_rng(1).uniform(0.3, 1.0, len(h))
    rng = np.random.default_rng(2)
    t0 = theta + rng.normal(0, 0.05, theta.size)
    f = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, dc=True)[0]
    g = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, dc=True)[1]
    assert check_grad(f, g, t0, epsilon=1e-6) < 1e-4 * max(1.0, np.linalg.norm(g(t0)))


def test_tau_conserves_probability_mass():
    lam, mu = np.array([1.4, 0.7, 2.2]), np.array([1.1, 0.9, 0.6])
    for rho in (-0.15, 0.0, 0.1):
        joint = P._apply_tau(P._pmf_grid(lam, mu, 30), lam, mu, rho)
        np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1e-10)


def test_rho_zero_equals_independent_poisson():
    lam, mu = np.array([1.5, 0.8]), np.array([1.0, 1.2])
    np.testing.assert_allclose(P.outcome_probs(lam, mu, rho=0.0), P.outcome_probs(lam, mu))


def test_negative_rho_raises_draw_probability():
    lam, mu = np.array([1.4]), np.array([1.1])
    assert P.outcome_probs(lam, mu, rho=-0.15)[0, 1] > P.outcome_probs(lam, mu, rho=0.0)[0, 1]


def test_simulate_scores_matches_outcome_probs():
    n = 60000
    lam, mu = np.full(n, 1.4), np.full(n, 1.0)
    x, y = P.simulate_scores(lam, mu, np.random.default_rng(3), rho=-0.12)
    emp = np.array([np.mean(x > y), np.mean(x == y), np.mean(x < y)])
    np.testing.assert_allclose(emp, P.outcome_probs(lam[:1], mu[:1], rho=-0.12)[0], atol=0.01)


def test_recovers_gamma_and_rho():
    theta, h, a, x, y, X = _data(n=36000, seed=4, rho=-0.1, gamma=-0.15)
    # many repeats of a small league: enough data for tight estimates
    res = P.fit(h, a, x, y, np.ones(len(h)), T, 0.01, X=X, dc=True)
    gamma, rho = res.unpack_extra(T, 1, True)
    assert abs(gamma[0] - (-0.15)) < 0.03
    assert abs(rho - (-0.1)) < 0.03


def test_closed_doors_flag_boundaries(tmp_path):
    dates = ["2020-06-16", "2020-06-17", "2021-05-16", "2021-05-17", "2021-05-18"]
    pd.DataFrame({
        "Date": dates, "HomeTeam": ["A"] * 5, "AwayTeam": ["B"] * 5,
        "FTHG": 1, "FTAG": 0, "FTR": "H",
    }).to_csv(tmp_path / "m.csv", index=False)
    df, _ = W.load_matches(tmp_path / "m.csv")
    assert df["closed_doors"].tolist() == [0, 1, 1, 0, 0]


@pytest.mark.parametrize("spec", [W.BASE, W.ModelSpec("cd", closed_doors=True),
                                  W.ModelSpec("dc", dc=True), W.ModelSpec("C", True, True)])
def test_no_leakage_for_every_spec(spec):
    df, _ = synthetic_league(n_seasons=5, seed=5)
    df["closed_doors"] = (df["Date"] > pd.Timestamp("2019-01-01")).astype(int)
    start, end = "2019-08-01", "2020-08-01"
    p1 = W.walk_forward(df, 10, start, end, 0.5, 2.0, spec=spec)
    # corrupt every result at or after the first predicted block's date: earlier blocks must not change
    df2 = df.copy()
    cut = pd.Timestamp("2020-02-01")
    late = df2["Date"] >= cut
    df2.loc[late, ["FTHG", "FTAG"]] = df2.loc[late, ["FTAG", "FTHG"]].values + 3
    df2["y"] = np.where(df2.FTHG > df2.FTAG, 0, np.where(df2.FTHG == df2.FTAG, 1, 2))
    p2 = W.walk_forward(df2, 10, start, end, 0.5, 2.0, spec=spec)
    early = df["Date"] < cut
    idx = p1.index[early.loc[p1.index].values]
    np.testing.assert_allclose(p1.loc[idx].values, p2.loc[idx].values, atol=1e-7)

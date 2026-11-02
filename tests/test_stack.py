import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from src import stack as S


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 6, n)
    a = (h + rng.integers(1, 6, n)) % 6
    hg, ag = rng.poisson(1.5, n), rng.poisson(1.1, n)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    dates = pd.to_datetime("2020-08-01") + pd.to_timedelta(np.arange(n) * 3, "D")
    return pd.DataFrame({"Date": dates, "h": h, "a": a, "FTHG": hg.astype(float), "FTAG": ag.astype(float), "y": y})


def test_ordered_logit_probabilities_sum_to_one_and_respond_to_d():
    p = S.ol_probs([1.0, -0.4, 0.0], np.array([-300.0, 0.0, 300.0]))
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()
    assert p[0, 2] > p[1, 2] > p[2, 2]  # away win more likely when d is very negative
    assert p[2, 0] > p[1, 0] > p[0, 0]


def test_ordered_logit_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 150, 300)
    y = rng.integers(0, 3, 300)
    w = rng.uniform(0.5, 1.5, 300)
    th = np.array([0.8, -0.3, -0.2])
    _, g = S.ol_loglik_and_grad(th, d, y, w)
    num = approx_fprime(th, lambda t: S.ol_loglik_and_grad(t, d, y, w)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)


def test_ordered_logit_recovers_known_parameters():
    rng = np.random.default_rng(2)
    d = rng.normal(0, 150, 40000)
    true = np.array([1.2, -0.6, -0.3])
    p = S.ol_probs(true, d)
    y = (rng.random(len(d))[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    est = S.fit_ol(d, y)
    assert np.allclose(est, true, atol=0.08)


def test_elo_is_zero_sum_and_uses_only_earlier_matches():
    df = _df()
    d = S.elo_prematch(df, 20.0, 1.0)
    assert d[0] == 0.0
    df2 = df.copy()
    df2.loc[40:, ["FTHG", "FTAG"]] = [[5.0, 0.0]] * (len(df) - 40)
    df2["y"] = np.where(df2["FTHG"] > df2["FTAG"], 0, np.where(df2["FTHG"] == df2["FTAG"], 1, 2))
    d2 = S.elo_prematch(df2, 20.0, 1.0)
    assert np.allclose(d[:41], d2[:41])  # results from match 40 on cannot change d up to and including match 40


def test_elo_carry_pulls_ratings_back_after_a_gap():
    df = _df(n=30)
    df.loc[15:, "Date"] = df.loc[15:, "Date"] + pd.Timedelta(days=120)
    d1 = S.elo_prematch(df, 30.0, 1.0)
    d0 = S.elo_prematch(df, 30.0, 0.0)
    assert d0[15] == 0.0  # everything reset to 1500 after the gap
    assert abs(d1[15]) >= 0.0 and np.allclose(d1[:15], d0[:15])


def test_pools_reduce_to_the_first_model():
    rng = np.random.default_rng(3)
    p1 = rng.dirichlet([3, 2, 3], 50)
    p2 = rng.dirichlet([3, 2, 3], 50)
    assert np.allclose(S.linear_pool(p1, p2, 0.0), p1)
    assert np.allclose(S.log_pool(p1, p2, 1.0, 0.0), p1)
    assert np.allclose(S.log_pool(p1, p2, 0.5, 0.5).sum(axis=1), 1)


def test_log_pool_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    p1 = rng.dirichlet([3, 2, 3], 200)
    p2 = rng.dirichlet([3, 2, 3], 200)
    y = rng.integers(0, 3, 200)
    th = np.array([0.9, 0.2])
    _, g = S.log_pool_loglik_and_grad(th, p1, p2, y)
    num = approx_fprime(th, lambda t: S.log_pool_loglik_and_grad(t, p1, p2, y)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)

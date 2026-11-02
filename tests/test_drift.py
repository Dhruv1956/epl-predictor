import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from src import drift as D
from src import poisson as P


def _data(T=5, n=300, nP=4, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    p = np.sort(rng.integers(0, nP, n))
    th = np.concatenate([[0.1, 0.2], rng.normal(0, 0.2, 2 * nP * T)])
    _, _, A, B = D.unpack(th, T, nP)
    lam = np.exp(0.3 + A[p, h] - B[p, a])
    mu = np.exp(0.1 + A[p, a] - B[p, h])
    return h, a, rng.poisson(lam), rng.poisson(mu), np.ones(n), p, T, nP, th


def test_gradient_matches_finite_differences():
    h, a, x, y, w, p, T, nP, th = _data()
    _, g = D.loglik_and_grad(th, h, a, x, y, w, p, T, nP, 5.0, 12.0)
    num = approx_fprime(th, lambda t: D.loglik_and_grad(t, h, a, x, y, w, p, T, nP, 5.0, 12.0)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)


def test_one_period_equals_constant_strength_model_b():
    h, a, x, y, w, p, T, nP, _ = _data(nP=1)
    ref = P.fit(h, a, x, y, w, T, 5.0).theta
    got, _ = D.fit_drift(h, a, x, y, w, np.zeros_like(p), T, 5.0, 7.0)
    assert np.allclose(ref, got, atol=1e-4)


def test_stiff_prior_ties_the_periods_together():
    h, a, x, y, w, p, T, nP, _ = _data()
    th, _ = D.fit_drift(h, a, x, y, w, p, T, 5.0, 1e4)
    _, _, A, _ = D.unpack(th, T, nP)
    assert np.abs(A - A[0]).max() < 0.02


def test_period_index_is_compressed_and_monotone():
    d = pd.to_datetime(["2020-01-01", "2020-01-20", "2020-06-01", "2020-06-02", "2021-01-01"])
    p = D.period_index(d, "2020-01-01", 91)
    assert list(p) == [0, 0, 1, 1, 2]


def test_forecast_does_not_use_later_results():
    rng = np.random.default_rng(4)
    T, n = 6, 360
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 360, n)), "D")
    h = rng.integers(0, T, n)
    df = pd.DataFrame({"Date": dates, "h": h, "a": (h + rng.integers(1, T, n)) % T,
                       "FTHG": rng.poisson(1.4, n), "FTAG": rng.poisson(1.1, n)}).sort_values("Date").reset_index(drop=True)
    cut = pd.Timestamp("2020-10-01")
    a = D.walk_forward_drift(df, T, "2020-09-01", cut, 5.0, 10.0)
    df2 = df.copy()
    late = df2["Date"] >= cut
    df2.loc[late, ["FTHG", "FTAG"]] = rng.poisson(3, (late.sum(), 2))
    b = D.walk_forward_drift(df2, T, "2020-09-01", cut, 5.0, 10.0)
    assert np.allclose(a.values, b.values)


def test_covariate_gradient_and_one_block_equals_model_b_with_covariates():
    h, a, x, y, w, p, T, nP, th = _data(nP=3)
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, (len(h), 2))
    Xm = rng.normal(0, 1, (len(h), 2))
    th = np.concatenate([th, [0.1, -0.2]])
    _, g = D.loglik_and_grad(th, h, a, x, y, w, p, T, nP, 5.0, 12.0, X, Xm)
    num = approx_fprime(th, lambda t: D.loglik_and_grad(t, h, a, x, y, w, p, T, nP, 5.0, 12.0, X, Xm)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)
    p0 = np.zeros_like(p)
    ref = P.fit(h, a, x, y, w, T, 5.0, X=X, Xm=Xm).theta
    got, _ = D.fit_drift(h, a, x, y, w, p0, T, 5.0, 7.0, X=X, Xm=Xm)
    assert np.allclose(ref, got, atol=1e-4)

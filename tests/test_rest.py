import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src import poisson as P
from src import rest as R


def _df():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-01", "2026-08-01", "2026-08-04", "2026-08-08", "2026-08-08"]),
        "HomeTeam": ["A", "C", "B", "A", "C"],
        "AwayTeam": ["B", "D", "A", "D", "B"],
    })
    return d


def test_rest_days_use_only_earlier_matches():
    d = _df()
    rh, ra = R.rest_days(d)
    # row 2: B at home on 4 Aug, B last played 1 Aug (3 days); A (away) last played 1 Aug (3 days)
    assert (rh[2], ra[2]) == (3, 3)
    # row 3: A home on 8 Aug, last played 4 Aug (4 days); D (away) last played 1 Aug (7 days)
    assert (rh[3], ra[3]) == (4, 7)
    # row 4: C home on 8 Aug, last played 1 Aug (7); B away, last played 4 Aug (4)
    assert (rh[4], ra[4]) == (7, 4)
    # first match of a team has the default
    assert rh[0] == R.DEFAULT_REST and ra[0] == R.DEFAULT_REST


def test_rest_of_a_match_does_not_depend_on_later_matches():
    d = _df()
    rh, ra = R.rest_days(d)
    d2 = d.copy()
    d2.loc[4, "HomeTeam"] = "Z"  # change the last match
    rh2, ra2 = R.rest_days(d2)
    assert np.array_equal(rh[:4], rh2[:4]) and np.array_equal(ra[:4], ra2[:4])


def test_features():
    rh, ra = np.array([2.0, 7.0, 10.0]), np.array([7.0, 2.0, 10.0])
    assert np.allclose(R.feature(rh, ra, "linear"), [-5 / 3, 5 / 3, 0])
    assert np.allclose(R.feature(rh, ra, "short"), [1, -1, 0])


def test_covariate_gradient_for_the_shared_coefficient():
    rng = np.random.default_rng(0)
    T, n = 5, 200
    theta = P.random_theta(T, rng)
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    z = rng.normal(0, 1, (n, 1))
    X, Xm = z, -z
    gamma = np.array([0.1])
    th = np.concatenate([theta, gamma])
    lam, mu = P.rates(th, h, a, T, X, Xm)
    x, y = rng.poisson(lam), rng.poisson(mu)
    w = np.ones(n)
    _, g = P.loglik_and_grad(th, h, a, x, y, w, T, 5.0, X=X, Xm=Xm)
    num = approx_fprime(th, lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 5.0, X=X, Xm=Xm)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)

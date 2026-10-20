"""Tests for the newly-promoted-team offsets (Xm covariates and the newcomer flags)."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src import poisson as P
from src import walkforward as W
from tests.test_walkforward import synthetic_league

PROMO = W.ModelSpec("promo", closed_doors=False, promoted=True)


def test_gradient_with_xm_matches_finite_differences():
    T = 6
    rng = np.random.default_rng(0)
    h, a = P.round_robin_fixtures(T, 3)
    n = len(h)
    hn, an = (rng.random(n) < 0.2).astype(float), (rng.random(n) < 0.2).astype(float)
    X = np.column_stack([hn, -an])
    Xm = np.column_stack([an, -hn])
    theta = np.concatenate([P.random_theta(T, rng, sd=0.3), [-0.2, -0.15]])
    lam, mu = P.rates(theta, h, a, T, X, Xm)
    x, y = rng.poisson(lam), rng.poisson(mu)
    w = rng.uniform(0.3, 1.0, n)
    t0 = theta + rng.normal(0, 0.05, theta.size)
    f = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, Xm=Xm)[0]
    g = lambda t: P.loglik_and_grad(t, h, a, x, y, w, T, 2.0, X=X, Xm=Xm)[1]
    assert check_grad(f, g, t0, epsilon=1e-6) < 1e-4 * max(1.0, np.linalg.norm(g(t0)))


def test_xm_without_x_is_rejected():
    theta = np.zeros(P.n_params(3))
    with pytest.raises(ValueError):
        P.linear_predictors(theta, np.array([0]), np.array([1]), 3, None, np.zeros((1, 1)))


def test_newcomer_flags(tmp_path):
    rows = []
    # season "A": teams X, Y, Z; season "B": X, Y, W (W is new); season "C": X, W, V (V is new, Y left)
    for season, date, teams in (("A", "2016-09-01", "XYZ"), ("B", "2017-09-01", "XYW"), ("C", "2018-09-01", "XWV")):
        rows.append((season, date, teams[0], teams[1]))
        rows.append((season, date, teams[1], teams[2]))
    pd.DataFrame(rows, columns=["Season", "Date", "HomeTeam", "AwayTeam"]).assign(FTHG=1, FTAG=0, FTR="H").to_csv(
        tmp_path / "m.csv", index=False
    )
    df, _ = W.load_matches(tmp_path / "m.csv")
    got = {(r.Season, r.HomeTeam): r.h_new for r in df.itertuples()} | {(r.Season, r.AwayTeam): r.a_new for r in df.itertuples()}
    assert got[("A", "X")] == got[("A", "Y")] == got[("A", "Z")] == 0  # first season: nobody flagged
    assert got[("B", "W")] == 1 and got[("B", "X")] == got[("B", "Y")] == 0
    assert got[("C", "V")] == 1 and got[("C", "W")] == 0  # W is no longer new in its second season


def _promotion_league(n_seasons=14, seed=0, pi_a=-0.25, pi_b=-0.2):
    """Simulated league in which 3 clubs per season are replaced and the newcomers are weaker."""
    rng = np.random.default_rng(seed)
    T = 26
    alpha, beta = rng.normal(0, 0.25, T), rng.normal(0, 0.25, T)
    iota, eta = 0.2, 0.2
    members = list(range(20))
    out_rows = []
    for s in range(n_seasons):
        new = set()
        if s > 0:
            leave = rng.choice(members, 3, replace=False)
            pool = [t for t in range(T) if t not in members]
            join = rng.choice(pool, 3, replace=False)
            members = [t for t in members if t not in leave] + list(join)
            new = set(int(t) for t in join)
        for hh in members:
            for aa in members:
                if hh == aa:
                    continue
                a_h = alpha[hh] + (pi_a if hh in new else 0)
                a_a = alpha[aa] + (pi_a if aa in new else 0)
                b_h = beta[hh] + (pi_b if hh in new else 0)
                b_a = beta[aa] + (pi_b if aa in new else 0)
                lam = np.exp(iota + eta + a_h - b_a)
                mu = np.exp(iota + a_a - b_h)
                out_rows.append((s, hh, aa, rng.poisson(lam), rng.poisson(mu)))
    df = pd.DataFrame(out_rows, columns=["s", "h", "a", "FTHG", "FTAG"])
    df["Season"] = [f"S{s:02d}" for s in df["s"]]
    df["Date"] = pd.to_datetime("2010-08-01") + pd.to_timedelta(df["s"] * 365 + np.random.default_rng(seed + 1).integers(0, 280, len(df)), unit="D")
    df["HomeTeam"] = [f"T{t:02d}" for t in df["h"]]
    df["AwayTeam"] = [f"T{t:02d}" for t in df["a"]]
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG == df.FTAG, "D", "A"))
    return df[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]


def test_recovers_promoted_offsets(tmp_path):
    _promotion_league().to_csv(tmp_path / "m.csv", index=False)
    df, teams = W.load_matches(tmp_path / "m.csv")
    res = W.fit_before(df, df["Date"].max() + pd.Timedelta(days=1), len(teams), xi=0.0, kappa=0.5, spec=PROMO)
    gamma, _ = res.unpack_extra(len(teams), 2, False)
    assert abs(gamma[0] - (-0.25)) < 0.08
    assert abs(gamma[1] - (-0.2)) < 0.08


def test_promoted_spec_beats_baseline_when_promotion_effect_exists(tmp_path):
    _promotion_league(seed=3).to_csv(tmp_path / "m.csv", index=False)
    df, teams = W.load_matches(tmp_path / "m.csv")
    T = len(teams)
    start, end = "2020-08-01", "2023-08-01"
    ll = {}
    for spec in (W.BASE, PROMO):
        wf = W.walk_forward(df, T, start, end, 0.3, 2.0, spec=spec)
        ll[spec.name] = float(np.mean(-np.log(wf.values[np.arange(len(wf)), df.loc[wf.index, "y"].values])))
    assert ll["promo"] < ll[W.BASE.name]


def test_no_leakage_with_promoted_spec():
    df, _ = synthetic_league(n_seasons=5, seed=6)
    df["closed_doors"] = 0
    df["h_new"] = ((df["h"] < 2) & (df["Date"].dt.year % 2 == 0)).astype(int)
    df["a_new"] = ((df["a"] < 2) & (df["Date"].dt.year % 2 == 0)).astype(int)
    start, end = "2019-08-01", "2020-08-01"
    p1 = W.walk_forward(df, 10, start, end, 0.5, 2.0, spec=PROMO)
    df2 = df.copy()
    cut = pd.Timestamp("2020-02-01")
    late = df2["Date"] >= cut
    df2.loc[late, ["FTHG", "FTAG"]] = df2.loc[late, ["FTAG", "FTHG"]].values + 3
    df2["y"] = np.where(df2.FTHG > df2.FTAG, 0, np.where(df2.FTHG == df2.FTAG, 1, 2))
    p2 = W.walk_forward(df2, 10, start, end, 0.5, 2.0, spec=PROMO)
    idx = p1.index[(df["Date"] < cut).loc[p1.index].values]
    np.testing.assert_allclose(p1.loc[idx].values, p2.loc[idx].values, atol=1e-7)


def test_fixed_parameters_are_held():
    T = 5
    rng = np.random.default_rng(0)
    theta = np.concatenate([P.random_theta(T, rng, sd=0.2), [0.1]])
    h, a = P.round_robin_fixtures(T, 6)
    X = (rng.random((len(h), 1)) < 0.3).astype(float)
    lam, mu = P.rates(theta, h, a, T, X)
    x, y = rng.poisson(lam), rng.poisson(mu)
    idx = 2 + 2 * T
    res = P.fit(h, a, x, y, np.ones(len(h)), T, 1.0, X=X, fixed={idx: -0.4})
    assert res.theta[idx] == pytest.approx(-0.4)
    free = P.fit(h, a, x, y, np.ones(len(h)), T, 1.0, X=X)
    assert res.loglik <= free.loglik + 1e-9  # constraining cannot raise the maximum


def test_two_stage_offsets_use_slow_decay_and_stay_fixed(tmp_path):
    _promotion_league(seed=5).to_csv(tmp_path / "m.csv", index=False)
    df, teams = W.load_matches(tmp_path / "m.csv")
    T = len(teams)
    ref = df["Date"].max() + pd.Timedelta(days=1)
    spec = W.ModelSpec("two-stage", promoted=True, promo_xi=0.0)
    two = W.fit_before(df, ref, T, xi=1.5, kappa=2.0, spec=spec)
    # stage-1 offsets: fitted with no decay at all
    stage1 = W.fit_before(df, ref, T, xi=0.0, kappa=2.0, spec=PROMO)
    np.testing.assert_allclose(two.unpack_extra(T, 2, False)[0], stage1.unpack_extra(T, 2, False)[0])


def test_two_stage_has_no_leakage():
    df, _ = synthetic_league(n_seasons=5, seed=8)
    df["closed_doors"] = 0
    df["h_new"] = ((df["h"] < 2) & (df["Date"].dt.year % 2 == 0)).astype(int)
    df["a_new"] = ((df["a"] < 2) & (df["Date"].dt.year % 2 == 0)).astype(int)
    spec = W.ModelSpec("two-stage", promoted=True, promo_xi=0.0)
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

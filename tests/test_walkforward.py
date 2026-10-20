"""Tests for src/walkforward.py, above all that there is no leakage from the future."""

import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src import poisson as P
from src import walkforward as W

T = 10


def synthetic_league(n_seasons=8, seed=0, sd=0.5):
    """Simulated league with known parameters, laid out like the real match table.

    Each season is a double round robin shuffled into weekly rounds of T/2 matches.
    """
    rng = np.random.default_rng(seed)
    theta = P.random_theta(T, rng, sd=sd)
    h, a, hg, ag = P.simulate_matches(theta, T, n_seasons, rng)
    per_season = T * (T - 1)
    dates = np.empty(len(h), dtype="datetime64[D]")
    for s in range(n_seasons):
        sl = slice(s * per_season, (s + 1) * per_season)
        perm = rng.permutation(per_season)
        h[sl], a[sl], hg[sl], ag[sl] = h[sl][perm], a[sl][perm], hg[sl][perm], ag[sl][perm]
        week = np.arange(per_season) // (T // 2)
        dates[sl] = np.datetime64(f"{2016 + s}-08-06") + 7 * week
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "h": h, "a": a, "FTHG": hg, "FTAG": ag, "y": y})
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    return df, theta


def test_no_leakage_from_future_results():
    df, _ = synthetic_league(n_seasons=6, seed=1)
    start, end = "2019-08-01", "2021-08-01"
    cutoff = pd.Timestamp("2020-03-01")

    # Scramble the goals of every match on or after the cutoff.
    df2 = df.copy()
    rng = np.random.default_rng(99)
    late = df2["Date"] >= cutoff
    df2.loc[late, "FTHG"] = rng.integers(0, 6, late.sum())
    df2.loc[late, "FTAG"] = rng.integers(0, 6, late.sum())

    wf1 = W.walk_forward(df, T, start, end, xi=0.3, kappa=2.0)
    wf2 = W.walk_forward(df2, T, start, end, xi=0.3, kappa=2.0)

    block_start = W.block_start_dates(df).loc[wf1.index]
    early = (block_start <= cutoff).values
    assert early.sum() > 50 and (~early).sum() > 50
    # Predictions for blocks that start on or before the cutoff only use matches before the
    # cutoff, which are identical in both tables, so the predictions must be identical.
    assert np.allclose(wf1.values[early], wf2.values[early], atol=1e-10, rtol=0)
    # Sanity check that the test has power: later predictions do react to the changed results.
    assert not np.allclose(wf1.values[~early], wf2.values[~early], atol=1e-3)


def test_every_window_match_is_predicted_once_with_valid_probabilities():
    df, _ = synthetic_league(n_seasons=5, seed=2)
    start, end = "2018-08-01", "2020-08-01"
    wf = W.walk_forward(df, T, start, end, xi=0.2, kappa=1.0)
    expected = df.index[(df["Date"] >= start) & (df["Date"] < end)]
    assert list(wf.index) == list(expected)
    assert np.allclose(wf.values.sum(axis=1), 1.0)
    assert (wf.values >= 0).all()


def test_walk_forward_learns_the_strengths():
    df, theta = synthetic_league(n_seasons=8, seed=3, sd=0.5)
    start, end = "2020-08-01", "2024-08-01"
    wf = W.walk_forward(df, T, start, end, xi=0.0, kappa=1.0)
    window = df.loc[wf.index]
    y = window["y"].values

    # Oracle: probabilities from the TRUE parameters (the best any model could do).
    lam, mu = P.rates(theta, window["h"].values, window["a"].values, T)
    oracle = M.log_loss(P.outcome_probs(lam, mu), y)
    model = M.log_loss(wf.values, y)
    base = M.log_loss(np.tile(M.class_frequencies(df[df["Date"] < start]["y"].values), (len(y), 1)), y)

    # The true parameters are the best model only in expectation, so on one finite sample a fitted
    # model can edge past them by chance. Beating them by a clear margin would signal leakage.
    assert model > oracle - 0.02
    assert model < base - 0.03  # the model has learned real structure
    assert model - oracle < 0.08  # and is not far from the best possible


def test_block_start_dates_are_the_first_date_of_each_week():
    df, _ = synthetic_league(n_seasons=2, seed=4)
    starts = W.block_start_dates(df)
    assert (starts <= df["Date"]).all()
    assert (df["Date"] - starts).max() <= pd.Timedelta(days=6)


def test_fit_before_requires_earlier_matches():
    df, _ = synthetic_league(n_seasons=2, seed=5)
    with pytest.raises(ValueError):
        W.fit_before(df, df["Date"].min(), T, xi=0.1, kappa=1.0)

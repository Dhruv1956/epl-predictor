"""Tests for the round-by-round backtest (src/backtest_weeks.py) and its place on the website."""

import json

import numpy as np
import pandas as pd
import pytest

from src import backtest_weeks as B
from src import live as L
from src import site_data as SD
from tests.test_live import league  # noqa: F401  (pytest fixture)
from tests.test_site import _freeze


def _write_full(tmp, full):
    path = tmp / "matches.csv"
    full.to_csv(path, index=False)
    return path


def _last_rounds(path, n=2):
    df, _ = L.load_matches(path)
    cur = df[df["Season"] == df["Season"].iloc[-1]].sort_values("Date")
    r = B.round_numbers(cur)
    top = int(r.max())
    return {top - i for i in range(n)}, cur.assign(round=r)


def test_training_set_is_exactly_the_matches_before_each_round(league):
    tmp, full, *_ = league
    path = _write_full(tmp, full)
    rounds, cur = _last_rounds(path)
    all_dates = pd.to_datetime(full["Date"])
    for k in sorted(rounds):
        season, scored, log, expls, skipped = B.run(path, {k})   # one round at a time
        assert not skipped
        first = cur.loc[cur["round"] == k, "Date"].min()
        want = int((all_dates < first).sum())                    # everything strictly before the round, nothing else
        assert set(log["n_train"]) == {want}, (k, want, set(log["n_train"]))
        assert pd.to_datetime(log["last_train_date"]).max() < first
        assert list(log.columns) == L.LOG_COLUMNS
        p = log[["pH", "pD", "pA"]].values
        assert np.all(p > 0) and np.allclose(p.sum(axis=1), 1.0)
        assert len(expls) == len(log) // len(L.MODELS)


def test_results_inside_a_round_cannot_change_its_predictions(league):
    """If the backtest peeked at the round it predicts, flipping scores in that round would move the probabilities."""
    tmp, full, *_ = league
    path = _write_full(tmp, full)
    rounds, cur = _last_rounds(path, 1)
    k = next(iter(rounds))
    _, _, log1, _, _ = B.run(path, {k})
    altered = full.copy()
    ids = cur.index[cur["round"] == k]
    dates = set(cur.loc[ids, "Date"])
    mask = pd.to_datetime(altered["Date"]).isin(dates)
    altered.loc[mask, "FTHG"] = altered.loc[mask, "FTHG"] + 3          # absurd scores
    altered.loc[mask, "FTR"] = "H"
    altered.loc[mask, "HST"] = altered.loc[mask, "HST"] + 9
    alt_path = tmp / "alt" / "matches.csv"
    alt_path.parent.mkdir()
    altered.to_csv(alt_path, index=False)
    _, _, log2, _, _ = B.run(alt_path, {k})
    cols = ["model_version", "home", "away", "pH", "pD", "pA", "lam", "mu"]
    a = log1[cols].sort_values(["model_version", "home", "away"]).reset_index(drop=True)
    b = log2[cols].sort_values(["model_version", "home", "away"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10)


def test_scores_match_an_independent_calculation(league):
    tmp, full, *_ = league
    path = _write_full(tmp, full)
    rounds, cur = _last_rounds(path)
    _, scored, log, _, _ = B.run(path, rounds)
    res = B.score(scored)
    row = next(r for r in res if r["round"] == "all" and r["model"] == "v2-shots")
    g = log[log["model_version"] == "v2-shots"].copy()
    full = full.copy()
    full["match_date"] = pd.to_datetime(full["Date"]).dt.strftime("%Y-%m-%d")
    j = g.merge(full, left_on=["match_date", "home", "away"], right_on=["match_date", "HomeTeam", "AwayTeam"])
    assert len(j) == len(g)
    y = np.where(j["FTHG"] > j["FTAG"], 0, np.where(j["FTHG"] == j["FTAG"], 1, 2))
    p = j[["pH", "pD", "pA"]].values
    assert row["n"] == len(y)
    assert row["log_loss"] == pytest.approx(-np.log(p[np.arange(len(y)), y]).mean(), abs=1e-12)
    assert row["brier"] == pytest.approx(((p - np.eye(3)[y]) ** 2).sum(axis=1).mean(), abs=1e-12)


def test_unknown_team_rounds_are_skipped_and_reported(league):
    tmp, full, *_ = league
    path = _write_full(tmp, full)
    df, _ = L.load_matches(path)
    cur = df[df["Season"] == df["Season"].iloc[-1]].sort_values("Date")
    r = B.round_numbers(cur)
    season, scored, log, expls, skipped = B.run(path, {int(r.min())})   # round 1: promoted sides unseen
    new_teams = set(cur["HomeTeam"]) - set(df[df["Season"] != df["Season"].iloc[-1]]["HomeTeam"])
    if new_teams:
        assert skipped and skipped[0]["round"] == int(r.min())
        assert set(skipped[0]["teams"]) <= new_teams | set(cur["AwayTeam"])
        assert scored.empty


def test_site_data_carries_the_backtest_and_scores_it_like_live(league, tmp_path):
    tmp, full, *_ = league
    log, expl, _ = _freeze(tmp)                       # a live log must exist for the page
    path = _write_full(tmp, full)
    rounds, _cur = _last_rounds(path)
    season, scored, blog, bexpl, skipped = B.run(path, rounds)
    out = tmp_path / "bt"
    B.write_outputs(out, season, blog, bexpl, skipped, sorted(rounds))
    d = SD.build_data(log, expl, path, backtest_dir=out)
    bt = d["backtest"]
    assert bt is not None and bt["rounds"] == sorted(rounds)
    n_matches = len(scored) // len(L.MODELS)
    assert sum(w["n"] for w in bt["weeks"]) == n_matches
    assert bt["scoreboard"]["n"] == n_matches
    assert all(m["explanation"] is not None for w in bt["weeks"] for m in w["matches"])
    json.dumps(d)                                      # NaN-free, serialisable
    assert SD.build_data(log, expl, path)["backtest"] is None   # absent unless asked for


def test_overlap_with_the_live_log_is_reported(league, tmp_path):
    """Matches in both the live log and the backtest are refitted from scratch; the page reports how closely they agree."""
    tmp, full, *_ = league
    log, expl, _ = _freeze(tmp)
    path = _write_full(tmp, full)
    rounds, _cur = _last_rounds(path)
    season, scored, blog, bexpl, skipped = B.run(path, rounds)
    out = tmp_path / "bt2"
    B.write_outputs(out, season, blog, bexpl, skipped, sorted(rounds))
    o = SD.build_data(log, expl, path, backtest_dir=out)["backtest"]["overlap"]
    assert o is not None and o["n"] > 0 and o["matchweeks"]
    assert np.isfinite(o["max_diff"]) and o["max_diff"] >= 0
    # identical inputs give identical probabilities: compare the live log with itself
    same = SD._overlap(log, log)
    assert same["max_diff"] == 0.0 and same["n"] == len(pd.read_csv(log)) // len(L.MODELS)

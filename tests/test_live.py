"""Tests for src/live.py: frozen predictions, kickoff filter, consistency with walk-forward, scoring."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import live as L
from src import walkforward as W
from tests.test_promoted import _promotion_league


@pytest.fixture
def league(tmp_path):
    """A 6-season league CSV. The last two calendar weeks become 'fixtures' (results hidden)."""
    full = _promotion_league(n_seasons=6, seed=2)
    full = full.sort_values("Date", kind="stable").reset_index(drop=True)
    cut = full["Date"].max() - pd.Timedelta(days=13)
    played, upcoming = full[full["Date"] < cut], full[full["Date"] >= cut]
    played.to_csv(tmp_path / "matches.csv", index=False)
    fx = upcoming[["Date", "HomeTeam", "AwayTeam"]].copy()
    fx["Date"] = fx["Date"].dt.strftime("%d/%m/%Y")
    fx.to_csv(tmp_path / "fixtures.csv", index=False)
    return tmp_path, full, upcoming, cut


def _run_predict(tmp, now):
    fixtures = L.read_fixtures(tmp / "fixtures.csv")
    return L.predict_fixtures(tmp / "matches.csv", fixtures, now), fixtures


def test_probabilities_are_valid(league):
    tmp, *_ = league
    rows, _ = _run_predict(tmp, datetime(2020, 1, 1))
    p = rows[["pH", "pD", "pA"]].values
    assert np.all(p > 0) and np.allclose(p.sum(axis=1), 1.0)
    assert set(rows["model_version"]) == set(L.MODELS)


def test_matches_walk_forward_prediction_for_the_same_block(league):
    tmp, full, upcoming, cut = league
    weeks = W.week_periods(upcoming["Date"])  # walk-forward refits at the start of each calendar week
    first_week = upcoming[weeks == weeks.min()]
    fx = pd.DataFrame({"Date": first_week["Date"], "HomeTeam": first_week["HomeTeam"], "AwayTeam": first_week["AwayTeam"]})
    fx["kickoff"] = pd.NaT
    rows = L.predict_fixtures(tmp / "matches.csv", fx.reset_index(drop=True), datetime(2020, 1, 1))
    v1 = rows[rows["model_version"] == "v1-working"][["pH", "pD", "pA"]].values

    full.to_csv(tmp / "full.csv", index=False)
    df, teams = W.load_matches(tmp / "full.csv")
    start, end = first_week["Date"].min(), first_week["Date"].max() + pd.Timedelta(days=1)
    wf = W.walk_forward(df, len(teams), start, end, 1.0, 5.0, spec=W.WORKING)
    keys = ["Date", "HomeTeam", "AwayTeam"]
    assert len(wf) == len(first_week) == len(v1)
    got = first_week[keys].reset_index(drop=True).assign(pH=v1[:, 0], pD=v1[:, 1], pA=v1[:, 2]).sort_values(keys)
    want = df.loc[wf.index, keys].assign(pH=wf["pH"], pD=wf["pD"], pA=wf["pA"]).sort_values(keys)
    np.testing.assert_allclose(got[["pH", "pD", "pA"]].values, want[["pH", "pD", "pA"]].values, atol=1e-4)


def test_predictions_are_frozen(league):
    tmp, *_ = league
    rows, _ = _run_predict(tmp, datetime(2020, 1, 1))
    log = tmp / "log.csv"
    added, frozen = L.append_frozen(rows, log)
    assert added == len(rows) and frozen == 0
    first = pd.read_csv(log)
    changed = rows.copy()
    changed["pH"] = 0.99  # a later run with different numbers must not overwrite
    added, frozen = L.append_frozen(changed, log)
    assert added == 0 and frozen == len(rows)
    pd.testing.assert_frame_equal(pd.read_csv(log), first)


def test_started_fixtures_are_dropped(league):
    tmp, *_ = league
    fx = L.read_fixtures(tmp / "fixtures.csv")
    mid = fx["Date"].iloc[len(fx) // 2]
    upcoming, started = L.drop_started(fx, datetime.combine(mid.date(), datetime.min.time()))
    assert (upcoming["Date"] >= mid).all() and (started["Date"] < mid).all() and len(started) > 0


def test_unknown_team_is_rejected(league):
    tmp, *_ = league
    fx = L.read_fixtures(tmp / "fixtures.csv")
    fx.loc[0, "HomeTeam"] = "Not A Club"
    with pytest.raises(ValueError, match="Not A Club"):
        L.predict_fixtures(tmp / "matches.csv", fx, datetime(2020, 1, 1))


def test_scoring_joins_results_and_bookmakers(league):
    tmp, full, upcoming, cut = league
    rows, _ = _run_predict(tmp, datetime(2020, 1, 1))
    rows[["bookH", "bookD", "bookA"]] = [0.4, 0.3, 0.3]  # pretend odds were recorded
    log = tmp / "log.csv"
    L.append_frozen(rows, log)
    full.to_csv(tmp / "matches.csv", index=False)  # results arrive
    card, joined = L.score_log(log, tmp / "matches.csv")
    assert set(card["model"]) == set(L.MODELS)
    assert (card["matches"] == len(upcoming)).all()
    y = joined[joined["model_version"] == "v1-working"]["y"].values
    p = joined[joined["model_version"] == "v1-working"][["pH", "pD", "pA"]].values
    expected = -np.mean(np.log(p[np.arange(len(y)), y]))
    assert card.loc[card["model"] == "v1-working", "log_loss"].iloc[0] == pytest.approx(expected)


def test_scoring_tolerates_moved_fixtures_and_falls_back_to_match_table_odds(league):
    tmp, full, upcoming, cut = league
    rows, _ = _run_predict(tmp, datetime(2020, 1, 1))  # no bookmaker odds were logged
    rows["match_date"] = (pd.to_datetime(rows["match_date"]) + pd.Timedelta(days=1)).dt.date.astype(str)
    log = tmp / "log.csv"
    L.append_frozen(rows, log)
    with_odds = full.copy()
    with_odds[["AvgH", "AvgD", "AvgA"]] = [2.0, 3.5, 3.8]
    with_odds.to_csv(tmp / "matches.csv", index=False)
    card, joined = L.score_log(log, tmp / "matches.csv")
    assert (card["matches"] == len(upcoming)).all()
    assert (card["book_matches"] == len(upcoming)).all()
    assert joined[["bookH", "bookD", "bookA"]].notna().all().all()

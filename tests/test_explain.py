"""Tests for src/explain.py and the odds helpers in src/live.py."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import explain as E
from src import live as L
from tests.test_live import league  # noqa: F401  (pytest fixture)


def _records(tmp):
    fixtures = L.read_fixtures(tmp / "fixtures.csv")
    return E.explain_fixtures(tmp / "matches.csv", fixtures), fixtures


def test_terms_add_up_to_the_logged_rates(league):
    tmp, *_ = league
    recs, fixtures = _records(tmp)
    rows = L.predict_fixtures(tmp / "matches.csv", fixtures, datetime(2020, 1, 1))
    log = rows[rows["model_version"] == E.MODEL].reset_index(drop=True)
    assert len(recs) == len(log)
    for r, (_, g) in zip(recs, log.iterrows()):
        assert (r["home"], r["away"]) == (g["home"], g["away"])
        assert abs(r["lam"] - g["lam"]) < 1e-9 and abs(r["mu"] - g["mu"]) < 1e-9
        assert abs(sum(r["terms_lam"].values()) - np.log(r["lam"])) < 1e-8
        assert abs(sum(r["terms_mu"].values()) - np.log(r["mu"])) < 1e-8


def test_ranks_are_valid_and_form_is_before_the_fixture(league):
    tmp, full, upcoming, cut = league
    recs, _ = _records(tmp)
    for r in recs:
        n = r["n_clubs"]
        for side in ("home", "away"):
            for v in r["ranks"][side].values():
                assert 1 <= v <= n
            assert len(r["form"][side]) == 6
            assert all(pd.Timestamp(m["date"]) < cut for m in r["form"][side])
        assert all(pd.Timestamp(m["date"]) < cut for m in r["h2h"])
        assert r["terms_mu"]["home"] == 0.0


def test_explanations_are_frozen_append_only(league, tmp_path):
    tmp, *_ = league
    recs, _ = _records(tmp)
    path = tmp_path / "ex.jsonl"
    assert E.append_explanations(recs, path) == (len(recs), 0)
    changed = [dict(r, lam=99.0) for r in recs]
    assert E.append_explanations(changed, path) == (0, len(recs))
    again = E.read_explanations(path)
    assert [r["lam"] for r in again] == [r["lam"] for r in recs]


def test_attach_odds_fills_only_missing_and_never_overwrites():
    fx = pd.DataFrame({"Date": pd.to_datetime(["2026-10-24", "2026-10-24"]), "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
                       "AvgH": [np.nan, 2.0], "AvgD": [np.nan, 3.0], "AvgA": [np.nan, 4.0]})
    src = pd.DataFrame({"Date": pd.to_datetime(["2026-10-25", "2026-10-24"]), "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
                        "AvgH": [1.5, 9.0], "AvgD": [4.0, 9.0], "AvgA": [6.0, 9.0]})
    out = L.attach_odds(fx, src)
    assert out.loc[0, ["AvgH", "AvgD", "AvgA"]].tolist() == [1.5, 4.0, 6.0]   # moved a day: still matched
    assert out.loc[1, ["AvgH", "AvgD", "AvgA"]].tolist() == [2.0, 3.0, 4.0]   # existing odds untouched
    assert L.has_odds(out).all()
    far = src.assign(Date=pd.to_datetime(["2026-12-25", "2026-12-25"]))
    assert not L.has_odds(L.attach_odds(fx.iloc[[0]], far)).iloc[0]

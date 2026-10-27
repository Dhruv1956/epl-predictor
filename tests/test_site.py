"""Tests for the website data layer (src/site_data.py) and page builder (src/site.py)."""

import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import explain as E
from src import live as L
from src import site as S
from src import site_data as SD
from tests.test_live import league  # noqa: F401  (pytest fixture)


def _freeze(tmp, with_odds=False):
    """Freeze predictions + explanations for the league fixture's upcoming matches. Returns (log, expl, fixtures)."""
    fixtures = L.read_fixtures(tmp / "fixtures.csv")
    if with_odds:
        fixtures["AvgH"], fixtures["AvgD"], fixtures["AvgA"] = 2.0, 3.5, 4.0
    rows = L.predict_fixtures(tmp / "matches.csv", fixtures, datetime(2020, 1, 1))
    log, expl = tmp / "log.csv", tmp / "expl.jsonl"
    L.append_frozen(rows, log)
    E.append_explanations(E.explain_fixtures(tmp / "matches.csv", fixtures), expl)
    return log, expl, fixtures


def test_pending_state_has_no_results_and_valid_probabilities(league):
    tmp, *_ = league
    log, expl, fixtures = _freeze(tmp)
    d = SD.build_data(log, expl, tmp / "matches.csv")
    ms = [m for w in d["weeks"] for m in w["matches"]]
    assert len(ms) == len(fixtures)
    assert d["scoreboard"]["n"] == 0 and all(m["result"] is None for m in ms)
    for m in ms:
        assert set(m["versions"]) == set(L.MODELS)
        for v in m["versions"].values():
            assert abs(sum(v["p"]) - 1) < 1e-9
        assert m["explanation"] is not None


def test_scoreboard_matches_an_independent_calculation(league):
    tmp, full, upcoming, cut = league
    log, expl, fixtures = _freeze(tmp, with_odds=True)
    full.to_csv(tmp / "matches.csv", index=False)  # results arrive
    d = SD.build_data(log, expl, tmp / "matches.csv")
    ms = [m for w in d["weeks"] for m in w["matches"] if m["result"]]
    assert len(ms) == len(upcoming) == d["scoreboard"]["n"]
    lg = pd.read_csv(log)
    for row in d["scoreboard"]["models"]:
        g = lg[lg["model_version"] == row["version"]]
        up = upcoming.assign(match_date=upcoming["Date"].dt.strftime("%Y-%m-%d"))
        res = up.merge(g, left_on=["match_date", "HomeTeam", "AwayTeam"], right_on=["match_date", "home", "away"])
        assert len(res) == len(upcoming)
        y = np.where(res["FTHG"] > res["FTAG"], 0, np.where(res["FTHG"] == res["FTAG"], 1, 2))
        p = res[["pH", "pD", "pA"]].values
        ll = -np.log(p[np.arange(len(y)), y]).mean()
        assert row["log_loss"] == pytest.approx(ll, abs=1e-9)
        brier = ((p - np.eye(3)[y]) ** 2).sum(axis=1).mean()
        assert row["brier"] == pytest.approx(brier, abs=1e-9)
        assert row["accuracy"] == pytest.approx((p.argmax(axis=1) == y).mean())
    # bookmaker benchmark: the constant odds we froze, margin removed
    book = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0]); book = book / book.sum()
    assert d["scoreboard"]["book"]["before_kickoff"] == len(ms)
    ys = [m["result"]["y"] for m in ms]
    assert d["scoreboard"]["book"]["log_loss"] == pytest.approx(np.mean([-math.log(book[y]) for y in ys]), abs=1e-9)


def test_odds_fall_back_to_the_match_table_and_are_labelled(league):
    tmp, full, upcoming, cut = league
    log, expl, _ = _freeze(tmp, with_odds=False)        # nothing frozen
    full = full.copy()
    full["AvgH"], full["AvgD"], full["AvgA"] = 2.0, 3.5, 4.0
    full.to_csv(tmp / "matches.csv", index=False)
    d = SD.build_data(log, expl, tmp / "matches.csv")
    ms = [m for w in d["weeks"] for m in w["matches"]]
    assert all(m["book"]["when"] == "after the match" for m in ms)
    assert d["scoreboard"]["book"]["before_kickoff"] == 0


def test_json_is_clean_and_page_embeds_it_safely(league, tmp_path):
    tmp, *_ = league
    log, expl, _ = _freeze(tmp)
    d = SD.build_data(log, expl, tmp / "matches.csv")
    text = json.dumps(d)
    assert "NaN" not in text and "Infinity" not in text
    page = S.render(d)
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    back = json.loads(m.group(1).replace("<\\/", "</"))
    assert back["weeks"][0]["n"] == d["weeks"][0]["n"]
    assert "__DATA__" not in page and "</script></script>" not in page
    # a team name that tries to break out of the script tag stays inert
    evil = json.loads(text)
    evil["weeks"][0]["matches"][0]["home"] = "</script><img src=x onerror=alert(1)>"
    out = S.render(evil)
    assert out.count('id="data"') == 1 and "</script><img" not in out


def test_matchweek_number_counts_games_already_played(league):
    tmp, full, upcoming, cut = league
    log, expl, _ = _freeze(tmp)
    d = SD.build_data(log, expl, tmp / "matches.csv")
    played = pd.read_csv(tmp / "matches.csv", parse_dates=["Date"])
    season = played["Season"].iloc[-1]
    cur = played[played["Season"] == season]
    for w in d["weeks"]:
        for m in w["matches"]:
            before = cur[cur["Date"] < pd.Timestamp(m["date"])]
            g = lambda t: ((before["HomeTeam"] == t) | (before["AwayTeam"] == t)).sum()
            assert m["matchweek"] == max(g(m["home"]), g(m["away"])) + 1


def test_score_groups_add_up_to_the_logged_result_probabilities(league):
    tmp, *_ = league
    log, expl, _ = _freeze(tmp)
    d = SD.build_data(log, expl, tmp / "matches.csv")
    for w in d["weeks"]:
        for m in w["matches"]:
            g, p = m["grid"], m["versions"][SD.HEADLINE]["p"]
            for key, want in zip("HDA", p):
                assert g[key]["p"] == pytest.approx(want, abs=1e-6)                 # same numbers the log froze
                parts = sum(t[2] for t in g[key]["top"]) + g[key]["other"]
                assert parts == pytest.approx(g[key]["p"], abs=1e-12)               # top scores + the rest = the total
                tops = [t[2] for t in g[key]["top"]]
                assert tops == sorted(tops, reverse=True)
            assert sum(g[k]["p"] for k in "HDA") == pytest.approx(1.0, abs=1e-6)
            assert g["all_top"][0] == g["best"]
            assert max(t[2] for k in "HDA" for t in g[k]["top"]) == pytest.approx(g["best"][2])


def test_likeliest_score_can_be_a_draw_while_a_win_is_the_likeliest_result():
    """Man United v Bournemouth (v2): expected goals 1.852 and 1.272. A win is 51% but 1-1 is the likeliest single score."""
    lam, mu = 1.852, 1.272
    g = SD.score_grid(lam, mu)
    assert g["best"][:2] == [1, 1] and g["best_key"] == "D"
    assert g["best"][2] == pytest.approx(math.exp(-(lam + mu)) * lam * mu, rel=1e-4)      # P(1-1) = e^-(l+m) * l * m
    assert g["H"]["p"] > g["A"]["p"] > g["D"]["p"]
    assert g["H"]["top"][0][:2] == [2, 1] and g["H"]["top"][0][2] < g["best"][2]          # no single win beats 1-1 ...
    assert g["H"]["p"] > g["D"]["p"] * 2                                                  # ... but the wins add up to far more
    assert g["H"]["p"] == pytest.approx(0.510, abs=1e-3) and g["D"]["p"] == pytest.approx(0.227, abs=1e-3)


def test_a_wrong_frozen_date_is_shown_as_the_real_played_date(league, tmp_path):
    """The hand-made fixture list once had a match a day out. The log stays as frozen, the page shows the played date."""
    tmp, full, upcoming, cut = league
    log, expl, _ = _freeze(tmp)
    full.to_csv(tmp / "matches.csv", index=False)          # results arrive
    lg = pd.read_csv(log)
    first = lg.iloc[0]
    wrong = lg.copy()
    mask = (wrong["home"] == first["home"]) & (wrong["away"] == first["away"])
    real = wrong.loc[mask, "match_date"].iloc[0]
    wrong.loc[mask, "match_date"] = (pd.Timestamp(real) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bad_log = tmp_path / "bad_log.csv"
    wrong.to_csv(bad_log, index=False)
    before = bad_log.read_bytes()
    d = SD.build_data(bad_log, expl, tmp / "matches.csv")
    ms = {(m["home"], m["away"]): m for w in d["weeks"] for m in w["matches"]}
    m = ms[(first["home"], first["away"])]
    assert m["date"] == real and m["date_logged"] != real                    # real date shown, logged one remembered
    assert d["scoreboard"]["n"] == len(upcoming)                             # still scored
    assert all("date_logged" not in x for k, x in ms.items() if k != (first["home"], first["away"]))
    assert bad_log.read_bytes() == before                                    # building the page never edits the log

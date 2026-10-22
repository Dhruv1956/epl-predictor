"""Everything the live website needs, as one JSON-friendly dict, built only from frozen files.

    build_data(log_path, explanations_path, matches_path)

Inputs: live/predictions_log.csv (frozen predictions), live/explanations.jsonl (frozen explanations) and the
match table (results). Nothing is predicted or refitted here: a prediction is read from the log exactly as
it was frozen, and results are joined on afterwards.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import live as L
from .explain import read_explanations
from .walkforward import load_matches

VERSIONS = list(L.MODELS)
LABELS = {"v0-poisson": "v0 Goals only", "v1-working": "v1 Goals + context", "v2-shots": "v2 Goals + shots"}
HEADLINE = "v2-shots"
NEEDED_MATCHES = 350  # about 2 standard errors for a 0.0165 gap (see the Checkpoint 2 report)


def clean(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python ones, for JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def _loss(p, y):
    return float(-math.log(max(p[y], 1e-15)))


def _brier(p, y):
    return float(sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(3)))


def matchweek_number(played, home, away, date):
    """Matchweek of a fixture = how many league games the busier of the two teams has already played this season, plus one."""
    season = played["Season"].iloc[-1] if "Season" in played.columns else None
    cur = played[played["Season"] == season] if season is not None else played
    before = cur[cur["Date"] < pd.Timestamp(date) - pd.Timedelta(days=0)]
    n = lambda t: int(((before["HomeTeam"] == t) | (before["AwayTeam"] == t)).sum())
    return max(n(home), n(away)) + 1


def build_data(log_path, explanations_path, matches_path, now=None):
    log = pd.read_csv(log_path)
    if log.empty:
        raise ValueError("The prediction log is empty: freeze some predictions first.")
    df, _ = load_matches(matches_path)
    played = df.dropna(subset=["FTHG", "FTAG"])
    expl = {(r["match_date"], r["home"], r["away"]): r for r in read_explanations(explanations_path)}
    _, joined = L.score_log(log_path, matches_path)  # pairs each frozen row with its result (+- 7 days)

    key = ["match_date", "home", "away"]
    log["match_date"] = log["match_date"].astype(str)
    joined = joined.copy()
    joined["match_date"] = joined["match_date"].dt.strftime("%Y-%m-%d")
    res_index = {(r.match_date, r.home, r.away, r.model_version): r for r in joined.itertuples()}

    matches = []
    for (date, home, away), g in log.groupby(key, sort=False):
        g = g.set_index("model_version")
        m = {
            "date": date, "home": home, "away": away, "kickoff": str(g["kickoff"].iloc[0]) if pd.notna(g["kickoff"].iloc[0]) else "",
            "logged_at": str(g["logged_at"].min()), "n_train": int(g["n_train"].iloc[0]), "last_train_date": str(g["last_train_date"].iloc[0]),
            "versions": {}, "explanation": expl.get((date, home, away)), "result": None, "book": None,
            "matchweek": matchweek_number(played, home, away, date),
        }
        for v in VERSIONS:
            if v in g.index:
                r = g.loc[v]
                m["versions"][v] = {"p": [r["pH"], r["pD"], r["pA"]], "lam": r["lam"], "mu": r["mu"]}
        # book odds frozen at prediction time (any version row carries the same values)
        frozen_book = g[["bookH", "bookD", "bookA"]].iloc[0].tolist()
        frozen_source = g["book_source"].iloc[0] if "book_source" in g.columns else None
        if all(pd.notna(x) for x in frozen_book):
            m["book"] = {"p": frozen_book, "source": str(frozen_source), "when": "before kickoff"}
        any_res = next((res_index[(date, home, away, v)] for v in VERSIONS if (date, home, away, v) in res_index), None)
        if any_res is not None:
            y = int(any_res.y)
            m["result"] = {"hg": int(any_res.FTHG), "ag": int(any_res.FTAG), "y": y, "played_on": any_res.Date.strftime("%Y-%m-%d")}
            if m["book"] is None and all(pd.notna(x) for x in (any_res.bookH, any_res.bookD, any_res.bookA)):
                m["book"] = {"p": [any_res.bookH, any_res.bookD, any_res.bookA], "source": "match table", "when": "after the match"}
            for v, d in m["versions"].items():
                d["loss"] = _loss(d["p"], y)
                d["brier"] = _brier(d["p"], y)
                d["hit"] = bool(int(np.argmax(d["p"])) == y)
            if m["book"] is not None:
                m["book"]["loss"] = _loss(m["book"]["p"], y)
                m["book"]["hit"] = bool(int(np.argmax(m["book"]["p"])) == y)
        matches.append(m)

    matches.sort(key=lambda m: (m["date"], m["kickoff"], m["home"]))

    # ---- weeks -------------------------------------------------------------------------------
    weeks = {}
    for m in matches:
        weeks.setdefault(m["matchweek"], []).append(m)
    week_list = []
    for mw in sorted(weeks):
        ms = weeks[mw]
        dates = sorted({m["date"] for m in ms})
        week_list.append({"matchweek": mw, "first": dates[0], "last": dates[-1], "n": len(ms),
                          "n_played": sum(1 for m in ms if m["result"]), "matches": ms})

    # ---- scoreboard --------------------------------------------------------------------------
    done = [m for m in matches if m["result"] and all(v in m["versions"] for v in VERSIONS)]
    board = {"n": len(done), "needed": NEEDED_MATCHES, "models": [], "book": None}
    for v in VERSIONS:
        if not done:
            break
        ll = [m["versions"][v]["loss"] for m in done]
        row = {"version": v, "label": LABELS[v], "log_loss": float(np.mean(ll)),
               "brier": float(np.mean([m["versions"][v]["brier"] for m in done])),
               "accuracy": float(np.mean([m["versions"][v]["hit"] for m in done]))}
        both = [m for m in done if m["book"] and "loss" in m["book"]]
        if len(both) >= 2:
            d = np.array([m["versions"][v]["loss"] - m["book"]["loss"] for m in both])
            row.update(gap=float(d.mean()), gap_se=float(d.std(ddof=1) / np.sqrt(len(d))), gap_n=len(both))
        board["models"].append(row)
    withbook = [m for m in done if m["book"] and "loss" in m["book"]]
    if withbook:
        board["book"] = {"n": len(withbook), "log_loss": float(np.mean([m["book"]["loss"] for m in withbook])),
                         "brier": float(np.mean([_brier(m["book"]["p"], m["result"]["y"]) for m in withbook])),
                         "accuracy": float(np.mean([m["book"]["hit"] for m in withbook])),
                         "before_kickoff": int(sum(1 for m in withbook if m["book"]["when"] == "before kickoff"))}
    now = now or datetime.now(timezone.utc)
    cfg = L.MODELS[HEADLINE]
    decay = {"goals_months": math.log(2) / cfg["xi"] * 12,
             "shots_months": math.log(2) / cfg["spec"].shots_xi * 12 if cfg["spec"].shots else None}
    return clean({
        "decay": decay,
        "generated_at": now.isoformat(timespec="minutes"), "season": str(played["Season"].iloc[-1]) if "Season" in played.columns else "",
        "data_through": played["Date"].max().strftime("%Y-%m-%d"), "n_train_latest": int(log["n_train"].max()),
        "headline": HEADLINE, "labels": LABELS, "versions": VERSIONS, "weeks": week_list, "scoreboard": board,
    })

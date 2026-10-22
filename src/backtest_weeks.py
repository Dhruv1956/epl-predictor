"""Round-by-round backtest on the current season.

For each round k of the latest season, every registered model is refitted using ONLY matches played before the
first kickoff of round k (all earlier seasons plus rounds 1..k-1), predicts round k, and is scored against the
actual results and against the bookmakers (margin removed; Avg odds, else B365).

This is a *backtest*, not the frozen live log: the code and hyperparameters were fixed before these rounds were
looked at, but the rounds were already played when the test was run, so it carries less weight than live results.

Outputs (all regenerated on every run, in --out-dir):
    predictions_log.csv   same columns as live/predictions_log.csv, so the website reads it with the same code
    explanations.jsonl    the "why" record for each match, built from the same pre-round data
    meta.json             season, rounds covered, rounds skipped and why
and reports/backtest_season.csv, the per-round score table.

    python -m src.backtest_weeks [--matches data/processed/matches.csv] [--out-dir backtest] [--rounds 5 6 7]
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .explain import explain_fixtures
from .live import MODELS, load_matches, predict_fixtures


def _book(row):
    for cols in (["AvgH", "AvgD", "AvgA"], ["B365H", "B365D", "B365A"]):
        if all(c in row.index and pd.notna(row[c]) for c in cols):
            return M.implied_probs(np.array([[row[c] for c in cols]], dtype=float))[0]
    return np.array([np.nan] * 3)


def round_numbers(cur):
    """Round of a match = most games either side has already played this season, plus one."""
    out = []
    for _, r in cur.iterrows():
        before = cur[cur["Date"] < r["Date"]]
        n = lambda t: int(((before["HomeTeam"] == t) | (before["AwayTeam"] == t)).sum())
        out.append(max(n(r["HomeTeam"]), n(r["AwayTeam"])) + 1)
    return pd.Series(out, index=cur.index)


def run(matches_path, only_rounds=None):
    """Returns (season, scored, log, explanations, skipped).

    scored: one row per (round, model, match) with the outcome, probabilities and bookmaker probabilities.
    log: concatenated predict_fixtures output (the live log's format). explanations: list of dicts.
    """
    raw = pd.read_csv(matches_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    shots_path = Path(matches_path).with_name("shots.csv")
    if "HST" not in raw.columns and shots_path.exists():
        shots = pd.read_csv(shots_path, parse_dates=["Date"])
        raw = raw.merge(shots, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    df, _ = load_matches(matches_path)
    played = df.dropna(subset=["FTHG", "FTAG"])
    season = played["Season"].iloc[-1]
    cur = played[played["Season"] == season].sort_values("Date").copy()
    cur["round"] = round_numbers(cur)
    scored, logs, expls, skipped = [], [], [], []
    for k, grp in cur.groupby("round"):
        if only_rounds is not None and int(k) not in only_rounds:
            continue
        ref = grp["Date"].min()
        now = (ref - pd.Timedelta(days=1)).to_pydatetime()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.csv"
            raw[raw["Date"] < ref].to_csv(path, index=False)  # nothing from this round or later
            fx = grp[["Date", "HomeTeam", "AwayTeam"]].copy()
            fx["kickoff"] = pd.NaT
            fx = fx.reset_index(drop=True)
            try:
                pred = predict_fixtures(path, fx, now)
                expl = explain_fixtures(path, fx, now)
            except ValueError as e:
                if "Teams not found" not in str(e):
                    raise
                teams = re.findall(r"'([^']+)'", str(e).split(":", 1)[-1])
                skipped.append({"round": int(k), "teams": teams})
                print(f"Round {k}: skipped (no earlier match in the table for {', '.join(teams)})")
                continue
        logs.append(pred)
        expls.extend(expl)
        for _, p in pred.iterrows():
            m = grp[(grp["HomeTeam"] == p["home"]) & (grp["AwayTeam"] == p["away"])].iloc[0]
            y = 0 if m["FTHG"] > m["FTAG"] else (1 if m["FTHG"] == m["FTAG"] else 2)
            scored.append({"round": int(k), "model": p["model_version"], "home": p["home"], "away": p["away"],
                           "y": y, "p": [p["pH"], p["pD"], p["pA"]], "book": _book(m), "n_train": p["n_train"]})
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    return season, pd.DataFrame(scored), log, expls, skipped


def score(df):
    """Per (round, model) log loss, Brier, accuracy; bookmakers as their own row."""
    out = []

    def one(sub, name, key):
        P = np.vstack(sub[key].values)
        y = sub["y"].values
        ok = ~np.isnan(P).any(axis=1)
        P, y = P[ok], y[ok]
        ll = -np.log(P[np.arange(len(y)), y])
        br = ((P - np.eye(3)[y]) ** 2).sum(axis=1)
        return {"model": name, "n": len(y), "log_loss": ll.mean(), "brier": br.mean(), "accuracy": (P.argmax(1) == y).mean(), "_ll": ll}

    for k, g in [("all", df)] + [(r, d) for r, d in df.groupby("round")]:
        for m in MODELS:
            out.append({"round": k, **one(g[g["model"] == m], m, "p")})
        out.append({"round": k, **one(g[g["model"] == list(MODELS)[0]], "bookmakers", "book")})
    return out


def write_outputs(out_dir, season, log, expls, skipped, rounds):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.to_csv(out / "predictions_log.csv", index=False)
    with open(out / "explanations.jsonl", "w") as f:  # regenerated each run, not an append-only record
        for r in expls:
            f.write(json.dumps(r, default=str) + "\n")
    meta = {"season": season, "rounds": sorted(rounds), "skipped": skipped,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", default="data/processed/matches.csv")
    ap.add_argument("--out", default="reports/backtest_season.csv")
    ap.add_argument("--out-dir", default="backtest")
    ap.add_argument("--rounds", type=int, nargs="*", default=None, help="only these rounds (default: all)")
    a = ap.parse_args(argv)
    season, df, log, expls, skipped = run(a.matches, set(a.rounds) if a.rounds else None)
    res = score(df)
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "_ll"} for r in res])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    write_outputs(a.out_dir, season, log, expls, skipped, sorted(df["round"].unique().tolist()))
    print(f"Season {season}: {df['round'].nunique()} rounds, {len(df)//len(MODELS)} matches\n")
    pd.set_option("display.width", 140)
    for r in ["all"] + sorted(set(tab["round"]) - {"all"}, key=int):
        print(f"--- round {r} ---")
        print(tab[tab["round"] == r].drop(columns="round").round(4).to_string(index=False))
    book = next(r for r in res if r["round"] == "all" and r["model"] == "bookmakers")
    print("\nPaired log-loss difference vs bookmakers (all rounds; positive = model worse), +-SE treating matches as independent:")
    for r in res:
        if r["round"] == "all" and r["model"] in MODELS:
            n = min(len(r["_ll"]), len(book["_ll"]))
            d = r["_ll"][:n] - book["_ll"][:n]
            print(f"  {r['model']:12s} {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(n):.4f}")


if __name__ == "__main__":
    main()

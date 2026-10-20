"""Live predictions: freeze probabilities for upcoming matches, then score them once results arrive.

    python -m src.live predict --fixtures data/live/fixtures.csv
    python -m src.live score

`predict` fits each registered model on every played match in the match table, predicts the fixtures,
and APPENDS the predictions to live/predictions_log.csv. A row is never overwritten: once a
(match, model version) pair is logged it is frozen. Fixtures that have already kicked off are skipped.
`score` joins the log to the results in the match table and compares each model version with the
bookmakers' odds that were recorded at prediction time.

Fixtures file: needs Date (dd/mm/yyyy), HomeTeam, AwayTeam; optional Div (only E0 is kept), Time
(HH:MM) and odds columns (AvgH/D/A or B365H/D/A). football-data.co.uk's fixtures.csv has this layout.
Team names must match the names in the match table.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P
from .walkforward import (
    BASE,
    WORKING,
    _covariates,
    add_closed_doors_flag,
    add_newcomer_flags,
    fit_before,
    load_matches,
    predict_block,
)

# Registered model versions. Never edit an existing entry: add a new version instead, so that the
# log always says exactly which model produced each prediction.
MODELS = {
    "v0-poisson": {"spec": BASE, "xi": 1.0, "kappa": 5.0},
    "v1-working": {"spec": WORKING, "xi": 1.0, "kappa": 5.0},
}
ODDS_PRIORITY = [("Avg", ["AvgH", "AvgD", "AvgA"]), ("B365", ["B365H", "B365D", "B365A"])]
LOG_COLUMNS = [
    "logged_at", "model_version", "xi", "kappa", "match_date", "kickoff", "home", "away",
    "lam", "mu", "pH", "pD", "pA", "n_train", "last_train_date",
    "book_source", "odds_H", "odds_D", "odds_A", "bookH", "bookD", "bookA",
]
DEFAULT_LOG = "live/predictions_log.csv"


def read_fixtures(path):
    """Read and tidy a fixtures file. Returns a DataFrame with Date, kickoff, HomeTeam, AwayTeam, odds."""
    fx = pd.read_csv(path, encoding="utf-8-sig")
    if "Div" in fx.columns:
        fx = fx[fx["Div"] == "E0"]
    missing = [c for c in ("Date", "HomeTeam", "AwayTeam") if c not in fx.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    fx = fx.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True)
    if "Time" in fx.columns:
        fx["kickoff"] = pd.to_datetime(fx["Date"].dt.strftime("%Y-%m-%d") + " " + fx["Time"].astype(str), errors="coerce")
    else:
        fx["kickoff"] = pd.NaT
    return fx.sort_values("Date").reset_index(drop=True)


def _pick_odds(row):
    for name, cols in ODDS_PRIORITY:
        if all(c in row.index for c in cols) and row[cols].notna().all():
            return name, row[cols].astype(float).values
    return "", None


def drop_started(fx, now):
    """Remove fixtures whose kickoff has passed (or, with no kickoff time, whose date is before today)."""
    started = np.where(fx["kickoff"].notna(), fx["kickoff"] <= now, fx["Date"] < pd.Timestamp(now.date()))
    return fx[~started].reset_index(drop=True), fx[started]


def predict_fixtures(matches_path, fixtures, now, models=MODELS):
    """Fit every registered model on all played matches and predict the fixtures. Returns log rows."""
    df, teams = load_matches(matches_path)
    played = df.dropna(subset=["FTHG", "FTAG"])
    T = len(teams)
    index = {t: i for i, t in enumerate(teams)}
    unknown = sorted((set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])) - set(index))
    if unknown:
        raise ValueError(f"Teams not found in the match table (check spelling): {unknown}")
    if fixtures["Date"].max() > played["Date"].max() + pd.Timedelta(days=120):
        raise ValueError("Fixtures are more than 120 days after the last played match: refresh the data first.")

    # Fixture rows carry the current season label so the promoted-team flags can be computed.
    fx = pd.DataFrame({
        "Date": fixtures["Date"], "HomeTeam": fixtures["HomeTeam"], "AwayTeam": fixtures["AwayTeam"],
        "FTHG": np.nan, "FTAG": np.nan, "FTR": "H",
    })
    if "Season" in df.columns:
        fx["Season"] = played["Season"].iloc[-1]
    combined = pd.concat([played.drop(columns=["h", "a", "y", "closed_doors", "h_new", "a_new"]), fx], ignore_index=True)
    combined["h"] = combined["HomeTeam"].map(index).astype(int)
    combined["a"] = combined["AwayTeam"].map(index).astype(int)
    combined["y"] = 0  # outcome placeholder, unused for prediction
    combined = add_newcomer_flags(add_closed_doors_flag(combined))
    block = combined.iloc[len(played):]

    ref = fixtures["Date"].min()
    rows = []
    for version, cfg in models.items():
        res = fit_before(combined, ref, T, cfg["xi"], cfg["kappa"], spec=cfg["spec"])
        probs = predict_block(res.theta, block, T, cfg["spec"])
        X, Xm = _covariates(block, cfg["spec"])
        lam, mu = P.rates(res.theta, block["h"].values, block["a"].values, T, X, Xm, cfg["spec"].kappa_home is not None)
        n_train = int((combined["Date"] < ref).sum())
        last_train = combined.loc[combined["Date"] < ref, "Date"].max()
        for i, (_, f) in enumerate(fixtures.iterrows()):
            source, odds = _pick_odds(f)
            book = M.implied_probs(odds[None, :])[0] if odds is not None else [np.nan] * 3
            rows.append({
                "logged_at": now.isoformat(timespec="seconds"), "model_version": version,
                "xi": cfg["xi"], "kappa": cfg["kappa"], "match_date": f["Date"].date().isoformat(),
                "kickoff": "" if pd.isna(f["kickoff"]) else f["kickoff"].isoformat(timespec="minutes"),
                "home": f["HomeTeam"], "away": f["AwayTeam"], "lam": lam[i], "mu": mu[i],
                "pH": probs[i, 0], "pD": probs[i, 1], "pA": probs[i, 2],
                "n_train": n_train, "last_train_date": last_train.date().isoformat(),
                "book_source": source,
                "odds_H": np.nan if odds is None else odds[0], "odds_D": np.nan if odds is None else odds[1],
                "odds_A": np.nan if odds is None else odds[2],
                "bookH": book[0], "bookD": book[1], "bookA": book[2],
            })
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def append_frozen(new_rows, log_path):
    """Append rows whose (match, model version) is not yet in the log. Returns (n_added, n_already_frozen)."""
    log_path = Path(log_path)
    key = ["match_date", "home", "away", "model_version"]
    if log_path.exists():
        old = pd.read_csv(log_path)
        seen = set(map(tuple, old[key].values))
        fresh = new_rows[[tuple(r) not in seen for r in new_rows[key].values]]
        out = pd.concat([old, fresh], ignore_index=True)
    else:
        fresh, out = new_rows, new_rows
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(log_path, index=False)
    return len(fresh), len(new_rows) - len(fresh)


def score_log(log_path, matches_path):
    """Join the log with played results. Returns (per-model scorecard DataFrame, joined rows)."""
    log = pd.read_csv(log_path)
    df, _ = load_matches(matches_path)
    played = df.dropna(subset=["FTHG", "FTAG"])[["Date", "HomeTeam", "AwayTeam", "y"]]
    log["match_date"] = pd.to_datetime(log["match_date"])
    joined = log.merge(played, left_on=["match_date", "home", "away"], right_on=["Date", "HomeTeam", "AwayTeam"])
    rows = []
    for version, g in joined.groupby("model_version"):
        y = g["y"].values.astype(int)
        p = g[["pH", "pD", "pA"]].values
        row = {"model": version, "matches": len(g), **M.scores(p, y)}
        has_book = g[["bookH", "bookD", "bookA"]].notna().all(axis=1).values
        if has_book.sum() >= 2:
            pb = g.loc[has_book, ["bookH", "bookD", "bookA"]].values
            lm = -np.log(np.clip(p[has_book][np.arange(has_book.sum()), y[has_book]], 1e-15, 1))
            lb = -np.log(np.clip(pb[np.arange(has_book.sum()), y[has_book]], 1e-15, 1))
            d = lm - lb
            row.update(book_matches=int(has_book.sum()), book_log_loss=lb.mean(), dLL_vs_book=d.mean(),
                       se=d.std(ddof=1) / np.sqrt(len(d)))
        rows.append(row)
    return pd.DataFrame(rows), joined


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "score"):
        sp = sub.add_parser(name)
        sp.add_argument("--matches", default="data/processed/matches.csv")
        sp.add_argument("--log", default=DEFAULT_LOG)
    sp = sub.choices["predict"]
    sp.add_argument("--fixtures", default="data/live/fixtures.csv")
    sp.add_argument("--allow-past", action="store_true", help="also log fixtures that already started (dry runs only)")
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        now = datetime.now()
        fx = read_fixtures(args.fixtures)
        upcoming, started = (fx, fx.iloc[0:0]) if args.allow_past else drop_started(fx, now)
        if len(started):
            print(f"Skipping {len(started)} fixture(s) that have already started.")
        if upcoming.empty:
            raise SystemExit("No upcoming fixtures to predict.")
        rows = predict_fixtures(args.matches, upcoming, now)
        added, frozen = append_frozen(rows, args.log)
        print(f"Logged {added} new prediction row(s); {frozen} already frozen (not changed). Log: {args.log}")
        show = rows[rows["model_version"] == "v1-working"][["match_date", "home", "away", "pH", "pD", "pA", "book_source", "bookH", "bookD", "bookA"]]
        print("\nv1-working predictions (probabilities; book* = bookmaker implied, margin removed):")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        if not Path(args.log).exists():
            raise SystemExit(f"{args.log} not found: run `predict` first.")
        card, joined = score_log(args.log, args.matches)
        if card.empty:
            raise SystemExit("No logged predictions have results yet.")
        print("Live scorecard (lower log loss / Brier is better; dLL_vs_book < 0 means better than the bookmakers):")
        print(card.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out = Path(args.log).with_name("scorecard.csv")
        card.to_csv(out, index=False)
        n = int(card["matches"].max())
        print(f"\nOnly {n} scored match(es) per model so far: differences are meaningless until a few hundred accumulate.")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

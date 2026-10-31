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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P
from .walkforward import (
    BASE,
    WORKING,
    ModelSpec,
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
    # v1 plus shots on target from earlier matches (stage-0 shot strengths feed the goal rates)
    "v2-shots": {
        "spec": ModelSpec("working + shots (xi_s=1.0)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=1.0),
        "xi": 1.5,
        "kappa": 5.0,
    },
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


def has_odds(fx):
    """Boolean Series: does each fixture row carry a complete set of odds?"""
    ok = pd.Series(False, index=fx.index)
    for _, cols in ODDS_PRIORITY:
        if all(c in fx.columns for c in cols):
            ok |= fx[cols].notna().all(axis=1)
    return ok


def attach_odds(fx, odds_fx, max_days=7):
    """Fill missing odds in `fx` from `odds_fx` (e.g. football-data's fixtures.csv) for the same home/away
    pair played within `max_days` of the same date. Existing odds are never overwritten."""
    fx = fx.copy()
    cols = [c for _, cs in ODDS_PRIORITY for c in cs]
    for c in cols:
        if c not in fx.columns:
            fx[c] = np.nan
    src = odds_fx[["Date", "HomeTeam", "AwayTeam"] + [c for c in cols if c in odds_fx.columns]]
    for i, row in fx.iterrows():
        if has_odds(fx.loc[[i]]).iloc[0]:
            continue
        m = src[(src["HomeTeam"] == row["HomeTeam"]) & (src["AwayTeam"] == row["AwayTeam"])
                & ((src["Date"] - row["Date"]).abs() <= pd.Timedelta(days=max_days))]
        if len(m):
            for c in cols:
                if c in m.columns and pd.notna(m.iloc[0][c]):
                    fx.at[i, c] = m.iloc[0][c]
    return fx


def drop_started(fx, now):
    """Remove fixtures whose kickoff has passed (or, with no kickoff time, whose date is before today)."""
    started = np.where(fx["kickoff"].notna(), fx["kickoff"] <= now, fx["Date"] < pd.Timestamp(now.date()))
    return fx[~started].reset_index(drop=True), fx[started]


def _prepare(matches_path, fixtures, models=MODELS):
    """Validate fixtures and build the combined (played + upcoming) frame every model is fitted on.

    Shared by `predict_fixtures` and the explanation script so that both use exactly the same data."""
    df, teams = load_matches(matches_path)
    played = df.dropna(subset=["FTHG", "FTAG"])
    T = len(teams)
    index = {t: i for i, t in enumerate(teams)}
    unknown = sorted((set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])) - set(index))
    if unknown:
        raise ValueError(f"Teams not found in the match table (check spelling): {unknown}")
    if any(cfg["spec"].shots for cfg in models.values()):
        if "HST" not in df.columns:
            raise ValueError("A shots model is registered but shots data is missing: run `python -m src.add_shots`.")
        recent = played[played["Date"] >= played["Date"].max() - pd.Timedelta(days=21)]
        if recent["HST"].isna().any():
            raise ValueError("Shots data is out of date for recent matches: run `python -m src.add_shots` after refreshing results.")
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
    return {"df": df, "teams": teams, "played": played, "T": T, "index": index, "combined": combined, "block": block, "ref": ref}


def predict_fixtures(matches_path, fixtures, now, models=MODELS, stamp=None):
    """Fit every registered model on all played matches and predict the fixtures. Returns log rows."""
    prep = _prepare(matches_path, fixtures, models)
    T, combined, block, ref = prep["T"], prep["combined"], prep["block"], prep["ref"]
    rows = []
    for version, cfg in models.items():
        res = fit_before(combined, ref, T, cfg["xi"], cfg["kappa"], spec=cfg["spec"])
        aux = getattr(res, "aux", None)
        probs = predict_block(res.theta, block, T, cfg["spec"], aux)
        X, Xm = _covariates(block, cfg["spec"], aux)
        lam, mu = P.rates(res.theta, block["h"].values, block["a"].values, T, X, Xm, cfg["spec"].kappa_home is not None)
        n_train = int((combined["Date"] < ref).sum())
        last_train = combined.loc[combined["Date"] < ref, "Date"].max()
        for i, (_, f) in enumerate(fixtures.iterrows()):
            source, odds = _pick_odds(f)
            book = M.implied_probs(odds[None, :])[0] if odds is not None else [np.nan] * 3
            rows.append({
                "logged_at": stamp or now.isoformat(timespec="seconds"), "model_version": version,
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
    """Append rows whose (match, model version) is not yet in the log. Returns (n_added, n_already_frozen).

    The log is append-only at the byte level: existing lines are never read back and rewritten (a pandas
    round trip can re-serialise old floats, e.g. in the 16th digit). Only the new lines are added at the end.
    """
    log_path = Path(log_path)
    key = ["match_date", "home", "away", "model_version"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        old = pd.read_csv(log_path)
        if list(old.columns) != list(new_rows.columns):
            raise ValueError(f"log columns {list(old.columns)} differ from the new rows {list(new_rows.columns)}")
        seen = set(map(tuple, old[key].values))
        fresh = new_rows[[tuple(r) not in seen for r in new_rows[key].values]]
        if len(fresh):
            with open(log_path, "rb") as fh:
                fh.seek(-1, 2)
                needs_newline = fh.read(1) not in (b"\n", b"\r")
            with open(log_path, "a", encoding="utf-8", newline="") as fh:
                if needs_newline:
                    fh.write("\n")
                fresh.to_csv(fh, index=False, header=False, lineterminator="\n")
    else:
        fresh = new_rows
        fresh.to_csv(log_path, index=False, lineterminator="\n")
    return len(fresh), len(new_rows) - len(fresh)


def score_log(log_path, matches_path):
    """Join the log with played results. Returns (per-model scorecard DataFrame, joined rows).

    A logged match is paired with the result of the same home/away fixture played within 7 days of the
    logged date (a fixture moved for TV or weather still counts; each pairing occurs once a season).
    If the log has no bookmaker odds for a match, the odds in the match table (Avg, else B365) are used:
    those are captured later than the prediction, so they are a slightly stricter benchmark.
    """
    log = pd.read_csv(log_path)
    df, _ = load_matches(matches_path)
    odds_cols = [c for _, cols in ODDS_PRIORITY for c in cols if c in df.columns]
    played = df.dropna(subset=["FTHG", "FTAG"])[["Date", "HomeTeam", "AwayTeam", "y", "FTHG", "FTAG"] + odds_cols]
    log["match_date"] = pd.to_datetime(log["match_date"])
    joined = log.merge(played, left_on=["home", "away"], right_on=["HomeTeam", "AwayTeam"])
    joined = joined[(joined["Date"] - joined["match_date"]).abs() <= pd.Timedelta(days=7)].copy()
    book = ["bookH", "bookD", "bookA"]
    for _, cols in ODDS_PRIORITY:
        if not all(c in joined.columns for c in cols):
            continue
        need = joined[book].isna().any(axis=1) & joined[cols].notna().all(axis=1)
        if need.any():
            joined.loc[need, book] = M.implied_probs(joined.loc[need, cols].values.astype(float))
    rows = []
    for version, g in joined.groupby("model_version"):
        y = g["y"].values.astype(int)
        p = g[["pH", "pD", "pA"]].values
        row = {"model": version, "matches": len(g), **M.scores(p, y)}
        has_book = g[book].notna().all(axis=1).values
        if has_book.sum() >= 2:
            pb = g.loc[has_book, book].values
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
    sp.add_argument("--odds-from", default=None, help="a football-data fixtures file whose odds are attached to fixtures lacking them")
    sp.add_argument("--require-odds", action="store_true", help="refuse to freeze fixtures that have no bookmaker odds")
    sp.add_argument("--explanations", default="live/explanations.jsonl", help="where the frozen per-match explanations go")
    sp.add_argument("--no-explain", action="store_true", help="skip writing explanations")
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        # football-data kickoff times are UK local time, so compare with the current UK time (not the Mac's clock)
        now_aware = datetime.now(ZoneInfo("Europe/London"))
        now = now_aware.replace(tzinfo=None)
        fx = read_fixtures(args.fixtures)
        upcoming, started = (fx, fx.iloc[0:0]) if args.allow_past else drop_started(fx, now)
        if len(started):
            print(f"Skipping {len(started)} fixture(s) that have already started.")
        if upcoming.empty:
            raise SystemExit("No upcoming fixtures to predict.")
        if args.odds_from:
            upcoming = attach_odds(upcoming, read_fixtures(args.odds_from))
        missing_odds = int((~has_odds(upcoming)).sum())
        if missing_odds:
            msg = (f"{missing_odds} of {len(upcoming)} fixture(s) have NO bookmaker odds. They will be frozen without a "
                   "pre-kickoff benchmark; scoring will fall back to the (slightly stricter) odds in the match table.")
            if args.require_odds:
                raise SystemExit(msg + " Aborting because of --require-odds.")
            print("WARNING: " + msg)
        rows = predict_fixtures(args.matches, upcoming, now, stamp=now_aware.isoformat(timespec="seconds"))
        added, frozen = append_frozen(rows, args.log)
        print(f"Logged {added} new prediction row(s); {frozen} already frozen (not changed). Log: {args.log}")
        if not args.no_explain:
            from .explain import append_explanations, explain_fixtures
            recs = explain_fixtures(args.matches, upcoming, now)
            a, f = append_explanations(recs, args.explanations)
            print(f"Explanations: {a} new, {f} already frozen. File: {args.explanations}")
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

"""Walk-forward validation for the Poisson model.

For each block of matches (one calendar week), the model is fitted using ONLY matches played
strictly before the block's first date, with exponential time-decay weights measured from that
date, and then predicts the whole block. Results inside a block are never used to predict it,
and nothing from the future is ever seen.

Expected columns in the match table (see load_matches):
    Date (datetime), HomeTeam, AwayTeam, FTHG, FTAG, FTR in {H, D, A}
and, added by load_matches: h, a (team indices 0..T-1) and y (outcome code 0/1/2).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def load_matches(path="data/processed/matches.csv"):
    """Read the cleaned match table. Returns (DataFrame sorted by date, list of team names)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if df["Date"].isna().any():
        raise ValueError("Some dates could not be parsed")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    index = {t: i for i, t in enumerate(teams)}
    df["h"] = df["HomeTeam"].map(index).astype(int)
    df["a"] = df["AwayTeam"].map(index).astype(int)
    df["y"] = df["FTR"].map(RESULT_CODE).astype(int)
    return df, teams


def week_periods(dates):
    """Calendar-week label (weeks end on Sunday) used to group matches into blocks."""
    return dates.dt.to_period("W-SUN")


def block_start_dates(df):
    """For each row, the first date of its weekly block."""
    return df.groupby(week_periods(df["Date"]))["Date"].transform("min")


def fit_before(df, ref_date, T, xi, kappa, theta0=None):
    """Fit the Poisson model on all matches strictly before ref_date, weights decayed from ref_date."""
    ref = pd.Timestamp(ref_date)
    train = df[df["Date"] < ref]
    if len(train) == 0:
        raise ValueError(f"No matches before {ref.date()}")
    age = P.age_in_years(train["Date"].values, ref.to_datetime64())
    w = P.time_weights(age, xi)
    return P.fit(
        train["h"].values,
        train["a"].values,
        train["FTHG"].values,
        train["FTAG"].values,
        w,
        T,
        kappa,
        theta0=theta0,
    )


def walk_forward(df, T, start, end, xi, kappa):
    """Predict every match with start <= Date < end, one weekly block at a time.

    Returns a DataFrame indexed like df with columns pH, pD, pA.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta = None  # warm start: the previous block's optimum is a good initial guess
    indices, probs = [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        block_start = block["Date"].min()
        res = fit_before(df, block_start, T, xi, kappa, theta0=theta)
        theta = res.theta
        lam, mu = P.rates(theta, block["h"].values, block["a"].values, T)
        probs.append(P.outcome_probs(lam, mu))
        indices.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(indices), columns=["pH", "pD", "pA"])
    return out.sort_index()


def grid_search(df, T, start, end, xis, kappas, on_result=None):
    """Walk-forward log loss on [start, end) for every (xi, kappa). Returns a DataFrame."""
    rows = []
    total = len(xis) * len(kappas)
    for xi in xis:
        for kappa in kappas:
            t0 = time.time()
            wf = walk_forward(df, T, start, end, xi, kappa)
            y = df.loc[wf.index, "y"].values
            row = {"xi": xi, "kappa": kappa, **M.scores(wf.values, y), "seconds": time.time() - t0}
            rows.append(row)
            if on_result is not None:
                on_result(len(rows), total, row)
    return pd.DataFrame(rows)


def team_strengths(theta, teams, T):
    """Fitted attack and defence strengths as a DataFrame (one row per team)."""
    _, _, alpha, beta = P.unpack(theta, T)
    return pd.DataFrame({"team": teams, "attack": alpha, "defence": beta})

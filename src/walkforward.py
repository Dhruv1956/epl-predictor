"""Walk-forward validation for the Poisson models.

For each block of matches (one calendar week), the model is fitted using ONLY matches played
strictly before the block's first date, with exponential time-decay weights measured from that
date, and then predicts the whole block. Results inside a block are never used to predict it,
and nothing from the future is ever seen.

Expected columns in the match table (see load_matches):
    Date (datetime), HomeTeam, AwayTeam, FTHG, FTAG, FTR in {H, D, A}
and, added by load_matches: h, a (team indices 0..T-1), y (outcome code 0/1/2) and
closed_doors (1 if the match was played without fans, else 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
RESULT_CODE = {"H": 0, "D": 1, "A": 2}

# Matches played without fans: [start, end) date windows.
#   * 2019/20 resumed behind closed doors on 17 June 2020 (after the March suspension).
#   * 2020/21 was played mostly without fans. Home supporters returned for the rounds from
#     17 May 2021 (18-19 May and 23 May), so the window ends there.
# A brief spell of limited crowds in December 2020 is not modelled (treated as closed doors).
CLOSED_DOORS_WINDOWS = [("2020-06-17", "2021-05-17")]


@dataclass(frozen=True)
class ModelSpec:
    """Which parts of the model are switched on."""

    name: str = "B: Poisson"
    closed_doors: bool = False  # separate home-advantage shift gamma for matches without fans
    dc: bool = False  # Dixon-Coles low-score correction (adds rho)


BASE = ModelSpec()


def add_closed_doors_flag(df):
    """Add a 0/1 column marking matches played without fans (see CLOSED_DOORS_WINDOWS)."""
    flag = np.zeros(len(df), dtype=int)
    for start, end in CLOSED_DOORS_WINDOWS:
        flag[((df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end))).values] = 1
    df["closed_doors"] = flag
    return df


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
    return add_closed_doors_flag(df), teams


def week_periods(dates):
    """Calendar-week label (weeks end on Sunday) used to group matches into blocks."""
    return dates.dt.to_period("W-SUN")


def block_start_dates(df):
    """For each row, the first date of its weekly block."""
    return df.groupby(week_periods(df["Date"]))["Date"].transform("min")


def _covariates(frame, spec):
    return frame[["closed_doors"]].values.astype(float) if spec.closed_doors else None


def fit_before(df, ref_date, T, xi, kappa, theta0=None, spec=BASE):
    """Fit on all matches strictly before ref_date, weights decayed from ref_date."""
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
        X=_covariates(train, spec),
        dc=spec.dc,
    )


def predict_block(theta, block, T, spec=BASE):
    """Outcome probabilities (n, 3) for a block of matches. Upcoming matches have fans (flag 0)."""
    X = _covariates(block, spec)
    lam, mu = P.rates(theta, block["h"].values, block["a"].values, T, X)
    k = 1 if spec.closed_doors else 0
    _, rho = P.unpack_extra(theta, T, k, spec.dc)
    return P.outcome_probs(lam, mu, rho=rho)


def walk_forward(df, T, start, end, xi, kappa, spec=BASE):
    """Predict every match with start <= Date < end, one weekly block at a time.

    Returns a DataFrame indexed like df with columns pH, pD, pA.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta = None  # warm start: the previous block's optimum is a good initial guess
    indices, probs = [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        res = fit_before(df, block["Date"].min(), T, xi, kappa, theta0=theta, spec=spec)
        theta = res.theta
        probs.append(predict_block(theta, block, T, spec))
        indices.append(block.index.values)
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(indices), columns=["pH", "pD", "pA"])
    return out.sort_index()


def grid_search(df, T, start, end, xis, kappas, on_result=None, spec=BASE):
    """Walk-forward log loss on [start, end) for every (xi, kappa). Returns a DataFrame."""
    rows = []
    total = len(xis) * len(kappas)
    for xi in xis:
        for kappa in kappas:
            t0 = time.time()
            wf = walk_forward(df, T, start, end, xi, kappa, spec=spec)
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

"""Rest-days feature for the Poisson model.

For each match, rest_h and rest_a are the days since that team's previous Premier League match in the data
(computed from earlier dates only, so a match's own result and everything after it are never used). Cup and European
matches are NOT in the data, so a team that played in midweek elsewhere still looks rested: the feature is a
lower bound on tiredness, and any effect it shows will be diluted.

Feature (linear):   z = ( min(rest_h, cap) - min(rest_a, cap) ) / scale
Feature (short):    z = 1[rest_h <= short] - 1[rest_a <= short]
Both enter the goal rates through ONE shared coefficient gamma:

    log lam += gamma * z        (the home side's goals rise if the home side is the more rested one)
    log mu  += -gamma * z

which is exactly the covariate form of Model B (X = z for log lam, Xm = -z for log mu, one gamma), so the gradient
is Model B's with the extra line  d ell / d gamma = sum_i w_i (z_i e_i - z_i f_i).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P
from .walkforward import week_periods

DEFAULT_REST = 14.0  # a team with no earlier match in the data (first match overall)


def rest_days(df):
    """(rest_h, rest_a) arrays aligned with df (sorted by date). Uses only each team's earlier matches."""
    last = {}
    rh = np.full(len(df), DEFAULT_REST)
    ra = np.full(len(df), DEFAULT_REST)
    # process matches in date order; two matches on the same date never affect each other's rest
    pos = {i: k for k, i in enumerate(df.index)}
    for date, grp in df.groupby("Date", sort=True):
        for i, h, a in zip(grp.index, grp["HomeTeam"], grp["AwayTeam"]):
            if h in last:
                rh[pos[i]] = (date - last[h]).days
            if a in last:
                ra[pos[i]] = (date - last[a]).days
        for h, a in zip(grp["HomeTeam"], grp["AwayTeam"]):
            last[h] = last[a] = date
    return rh, ra


def feature(rh, ra, kind="linear", cap=7.0, scale=3.0, short=3.0):
    if kind == "linear":
        return (np.minimum(rh, cap) - np.minimum(ra, cap)) / scale
    if kind == "short":
        return (rh <= short).astype(float) - (ra <= short).astype(float)
    raise ValueError(kind)


def walk_forward_rest(df, T, start, end, xi, kappa, z):
    """Weekly walk-forward with the rest feature z (array aligned with df). Returns probs and fitted gamma per block."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    zs = pd.Series(z, index=df.index)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    theta, idx, probs, gam = None, [], [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        ref = block["Date"].min()
        train = df[df["Date"] < ref]
        w = P.time_weights(P.age_in_years(train["Date"].values, ref.to_datetime64()), xi)
        zt = zs.loc[train.index].values[:, None]
        res = P.fit(train["h"].values, train["a"].values, train["FTHG"].values, train["FTAG"].values, w, T, kappa,
                    theta0=theta, X=zt, Xm=-zt)
        theta = res.theta
        zb = zs.loc[block.index].values[:, None]
        lam, mu = P.rates(theta, block["h"].values, block["a"].values, T, zb, -zb)
        probs.append(P.outcome_probs(lam, mu))
        idx.append(block.index.values)
        gam.append(float(theta[2 + 2 * T]))
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"]).sort_index()
    return out, np.array(gam)


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    from pathlib import Path
    from .walkforward import load_matches, walk_forward
    ap = argparse.ArgumentParser(description="Rest-days feature, walk-forward, Model B base.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--out", default="reports/rest")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)
    rh, ra = rest_days(df)
    desc = pd.Series(np.concatenate([rh, ra])).describe().round(2)
    print("rest days (all team-matches):", dict(desc))
    print("share of team-matches with rest <= 3 days: %.3f" % float((np.concatenate([rh, ra]) <= 3).mean()))
    base = {n: walk_forward(df, T, s, e, args.xi, args.kappa) for n, (s, e) in WINDOWS.items()}
    rows = []
    for kind in ["linear", "short"]:
        z = feature(rh, ra, kind)
        for n, (s, e) in WINDOWS.items():
            wf, gam = walk_forward_rest(df, T, s, e, args.xi, args.kappa, z)
            y = df.loc[wf.index, "y"].values
            ll_new, ll_old = -np.log(wf.values[np.arange(len(y)), y]), -np.log(base[n].values[np.arange(len(y)), y])
            d = ll_new - ll_old
            rows.append({"feature": kind, "window": n, "n": len(y), "LL_base": ll_old.mean(), "LL_rest": ll_new.mean(),
                         "delta": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)), "gamma_mean": gam.mean(), "gamma_last": gam[-1]})
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "rest_results.csv", index=False)


if __name__ == "__main__":
    main()

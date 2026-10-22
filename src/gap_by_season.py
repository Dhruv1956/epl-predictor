"""Per-season gap between plain Poisson (B) and the bookmakers, plus goal-rate drift.

    python -m src.gap_by_season [--data ...] [--out reports/gap_by_season.csv]

Walk-forward (weekly refits, fit only on matches before each week) with a registered live model (default v0-poisson: xi=1.0, kappa=5) for 2018/19..latest.
For each season: model log loss, B365 and Pinnacle (PS) log loss on the same matches, paired gaps +/- SE,
actual goals per match, previous-season goals per match, and mean predicted vs actual outcome frequencies.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from . import metrics as M
from .run_walkforward import ODDS_SOURCES
from .live import MODELS
from .walkforward import BASE, load_matches, walk_forward

def ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="v0-poisson", choices=sorted(MODELS), help="registered live model (spec, xi, kappa)")
    ap.add_argument("--first", default="2018/19")
    a = ap.parse_args(argv)
    cfg = MODELS[a.model]
    df, teams = load_matches(a.data)
    T = len(teams)
    df["goals"] = df["FTHG"] + df["FTAG"]
    seasons = sorted(df["Season"].unique())
    gpm = df.groupby("Season")["goals"].mean()
    rows = []
    for s in [x for x in seasons if x >= a.first]:
        sub = df[df["Season"] == s]
        start, end = sub["Date"].min().floor("D"), sub["Date"].max() + pd.Timedelta(days=1)
        p = walk_forward(df, T, start, end, cfg['xi'], cfg['kappa'], spec=cfg['spec'])
        fr = df.loc[p.index]
        y = fr["y"].values
        pm = p.values
        row = {"season": s, "n": len(y), "LL_model": ll(pm, y).mean(),
               "goals_pm": gpm[s], "goals_prev": gpm[seasons[seasons.index(s) - 1]],
               "home_pred": pm[:, 0].mean(), "home_act": (y == 0).mean(),
               "draw_pred": pm[:, 1].mean(), "draw_act": (y == 1).mean(),
               "away_pred": pm[:, 2].mean(), "away_act": (y == 2).mean()}
        for name, cols in (("B365", ODDS_SOURCES["B365"]), ("PS", ODDS_SOURCES["PS"]), ("Avg", ODDS_SOURCES["Avg"])):
            m = fr[cols].notna().all(axis=1).values
            if m.sum() < 50:
                continue
            lb = ll(M.implied_probs(fr.loc[m, cols].values.astype(float)), y[m])
            d = ll(pm[m], y[m]) - lb
            row[f"LL_{name}"] = lb.mean()
            row[f"gap_{name}"] = d.mean()
            row[f"se_{name}"] = d.std(ddof=1) / np.sqrt(m.sum())
            row[f"n_{name}"] = int(m.sum())
        rows.append(row)
        print(f"{s}: LL {row['LL_model']:.4f} gap B365 {row.get('gap_B365', float('nan')):+.4f} "
              f"PS {row.get('gap_PS', float('nan')):+.4f} goals {row['goals_pm']:.2f} (prev {row['goals_prev']:.2f})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out or f'reports/gap_by_season_{a.model}.csv', index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    for g in ("gap_B365", "gap_PS"):
        if g in out:
            o = out.dropna(subset=[g])
            print(f"\ncorr({g}, goals_pm - goals_prev) = {np.corrcoef(o[g], o.goals_pm - o.goals_prev)[0,1]:+.2f}  (n={len(o)} seasons)")
            print(f"corr({g}, |goals_pm - goals_prev|) = {np.corrcoef(o[g], (o.goals_pm - o.goals_prev).abs())[0,1]:+.2f}")

if __name__ == "__main__":
    main()

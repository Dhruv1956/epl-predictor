"""Tougher benchmark: closing odds, and the ranked probability score (RPS).

RPS respects that the outcomes are ordered H < D < A: with cumulative forecasts F_k = p_1 + ... + p_k and cumulative outcome
O_k = 1[y <= k] (k = 1, 2),
    RPS = 1/2 * sum_{k=1}^{2} (F_k - O_k)^2          (0 = perfect, 1 = confidently wrong by two steps)
Calling a draw a home win costs 1/2 of what calling a home win an away win costs; Brier and log loss treat both mistakes alike.

Bookmaker sources compared (margin removed by normalisation, as elsewhere in the project):
  Avg, PS, B365       odds recorded before the match (the benchmark used so far)
  AvgC, PSC, B365C    closing odds (the last prices before kickoff, the hardest benchmark; available from 2019/20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M

SOURCES = {"Avg": "Avg", "PS": "PS", "B365": "B365", "AvgC": "AvgC", "PSC": "PSC", "B365C": "B365C"}


def rps_per_match(probs, y):
    p = np.asarray(probs, float)
    o = np.eye(3)[np.asarray(y, int)]
    return 0.5 * ((np.cumsum(p, axis=1)[:, :2] - np.cumsum(o, axis=1)[:, :2]) ** 2).sum(axis=1)


def rps(probs, y):
    return float(np.mean(rps_per_match(probs, y)))


def load_closing(raw_dir="data/raw"):
    """Closing odds from the raw football-data files: Date, HomeTeam, AwayTeam, AvgC*, PSC*, B365C*."""
    from pathlib import Path
    from .data import parse_date
    frames = []
    for f in sorted(Path(raw_dir).glob("E0_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "FTR" not in df.columns:
            continue
        out = pd.DataFrame({"Date": parse_date(df["Date"]), "HomeTeam": df["HomeTeam"], "AwayTeam": df["AwayTeam"]})
        for src, pre in (("AvgC", "AvgC"), ("PSC", "PSC"), ("B365C", "B365C")):
            for o in "HDA":
                col = f"{pre}{o}"
                out[col] = df[col].values if col in df.columns else np.nan
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def book_probs(frame, name):
    cols = [f"{name}{o}" for o in "HDA"]
    ok = frame[cols].notna().all(axis=1) & (frame[cols] > 1.0).all(axis=1)
    p = np.full((len(frame), 3), np.nan)
    if ok.any():
        p[ok.values] = M.implied_probs(frame.loc[ok, cols].values)
    return p


WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}


def main(argv=None):
    import argparse
    from pathlib import Path
    from .live import MODELS
    from .walkforward import load_matches, walk_forward
    ap = argparse.ArgumentParser(description="Closing-odds benchmark and ranked probability score.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="reports/benchmark")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)
    df = df.merge(load_closing(args.raw), on=["Date", "HomeTeam", "AwayTeam"], how="left", suffixes=("", "_raw"))
    cfg = MODELS["v2-shots"]
    rows = []
    for w, (s, e) in WINDOWS.items():
        preds = {"Model B": walk_forward(df, T, s, e, 1.0, 5.0), "v2-shots": walk_forward(df, T, s, e, cfg["xi"], cfg["kappa"], spec=cfg["spec"])}
        sub = df.loc[preds["v2-shots"].index]
        y = sub["y"].values
        books = {k: book_probs(sub, k) for k in SOURCES}
        have = np.all([~np.isnan(b[:, 0]) for b in books.values()], axis=0)  # same matches for everyone
        if have.sum() == 0:
            continue
        allp = {**{k: v.values for k, v in preds.items()}, **{k + " odds": b for k, b in books.items()}}
        ref = {"ll": -np.log(allp["v2-shots"][have][np.arange(have.sum()), y[have]])}
        for name, p in allp.items():
            p, yy = p[have], y[have]
            llm = -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-15, 1))
            d = llm - ref["ll"]
            rows.append({"window": w, "n": int(have.sum()), "source": name, "log_loss": llm.mean(), "brier": M.brier_score(p, yy),
                         "rps": rps(p, yy), "LL_minus_v2": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)) if name != "v2-shots" else 0.0})
    res = pd.DataFrame(rows)
    res.to_csv(out / "benchmark.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

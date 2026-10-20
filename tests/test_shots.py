"""Tests for the shots-on-target extension."""

import numpy as np
import pandas as pd
import pytest

from src import add_shots as A
from src import walkforward as W

SHOTS = W.ModelSpec("shots", shots=True, shots_xi=0.3)


def shots_league(n_seasons=9, seed=0, c=0.8, T=20):
    """League where latent team quality drives shots on target, and goals follow that quality
    scaled by c plus a small team-specific finishing deviation."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 0.25, T), rng.normal(0, 0.25, T)
    f, g = rng.normal(0, 0.06, T), rng.normal(0, 0.06, T)
    rows = []
    for s in range(n_seasons):
        pairs = [(h, k) for h in range(T) for k in range(T) if h != k]
        rng.shuffle(pairs)
        for i, (h, k) in enumerate(pairs):
            date = pd.Timestamp(f"{2012 + s}-08-10") + pd.Timedelta(days=int(i // 10) * 6)
            sh = rng.poisson(np.exp(1.45 + 0.15 + a[h] - b[k]))
            sa = rng.poisson(np.exp(1.45 + a[k] - b[h]))
            gh = rng.poisson(np.exp(0.0 + 0.2 + c * (a[h] - b[k]) + f[h] - g[k]))
            ga = rng.poisson(np.exp(0.0 + c * (a[k] - b[h]) + f[k] - g[h]))
            rows.append((f"S{s}", date, f"T{h:02d}", f"T{k:02d}", gh, ga, sh, sa))
    df = pd.DataFrame(rows, columns=["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"])
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG == df.FTAG, "D", "A"))
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def _ll(wf, df):
    y = df.loc[wf.index, "y"].values
    return float(np.mean(-np.log(wf.values[np.arange(len(wf)), y])))


def test_recovers_the_shots_coefficient(tmp_path):
    shots_league(seed=1).to_csv(tmp_path / "m.csv", index=False)
    df, teams = W.load_matches(tmp_path / "m.csv")
    res = W.fit_before(df, df["Date"].max() + pd.Timedelta(days=1), len(teams), xi=0.0, kappa=2.0,
                       spec=W.ModelSpec("s", shots=True, shots_xi=0.0))
    gamma, _ = res.unpack_extra(len(teams), 1, False)
    assert abs(gamma[0] - 0.8) < 0.2


def test_shots_help_when_shots_carry_the_signal(tmp_path):
    shots_league(seed=2).to_csv(tmp_path / "m.csv", index=False)
    df, teams = W.load_matches(tmp_path / "m.csv")
    T = len(teams)
    start, end = "2018-08-01", "2020-08-01"
    base = _ll(W.walk_forward(df, T, start, end, 1.0, 5.0, spec=W.BASE), df)
    shots = _ll(W.walk_forward(df, T, start, end, 1.0, 5.0, spec=SHOTS), df)
    assert shots < base


def test_no_leakage_with_shots():
    df = shots_league(n_seasons=5, seed=3, T=10)
    df["Date"] = pd.to_datetime(df["Date"])
    df["h"] = df["HomeTeam"].str[1:].astype(int)
    df["a"] = df["AwayTeam"].str[1:].astype(int)
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    df["closed_doors"], df["h_new"], df["a_new"] = 0, 0, 0
    start, end = "2015-08-01", "2016-08-01"
    p1 = W.walk_forward(df, 10, start, end, 0.5, 2.0, spec=SHOTS)
    df2 = df.copy()
    cut = pd.Timestamp("2016-02-01")
    late = df2["Date"] >= cut
    df2.loc[late, ["HST", "AST", "FTHG", "FTAG"]] = df2.loc[late, ["AST", "HST", "FTAG", "FTHG"]].values + 5
    df2["y"] = np.where(df2.FTHG > df2.FTAG, 0, np.where(df2.FTHG == df2.FTAG, 1, 2))
    p2 = W.walk_forward(df2, 10, start, end, 0.5, 2.0, spec=SHOTS)
    idx = p1.index[(df["Date"] < cut).loc[p1.index].values]
    assert len(idx) > 50
    np.testing.assert_allclose(p1.loc[idx].values, p2.loc[idx].values, atol=1e-7)


def test_add_shots_and_automatic_merge(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "E0_2425.csv").write_text(
        "﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST\n"
        "E0,17/08/2024,A,B,2,1,H,12,8,5,3\nE0,18/08/24,B,A,0,0,D,9,10,2,4\n", encoding="utf-8")
    out = A.build_shots(raw)
    assert out["HST"].tolist() == [5, 2] and out["Date"].dt.year.tolist() == [2024, 2024]
    proc = tmp_path / "processed"
    proc.mkdir()
    out.drop(columns="season_file").to_csv(proc / "shots.csv", index=False)
    pd.DataFrame({"Season": ["2425"] * 2, "Date": ["2024-08-17", "2024-08-18"], "HomeTeam": ["A", "B"],
                  "AwayTeam": ["B", "A"], "FTHG": [2, 0], "FTAG": [1, 0], "FTR": ["H", "D"]}).to_csv(proc / "matches.csv", index=False)
    df, _ = W.load_matches(proc / "matches.csv")
    assert df.sort_values("Date")["HST"].tolist() == [5, 2]


def test_shots_spec_requires_stage0_strengths():
    block = pd.DataFrame({"h": [0], "a": [1], "closed_doors": [0], "h_new": [0], "a_new": [0]})
    with pytest.raises(ValueError):
        W._covariates(block, SHOTS)

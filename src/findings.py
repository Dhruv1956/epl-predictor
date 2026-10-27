"""Numbers for the website's Findings page.

Everything here is either read from a report CSV (season gaps, test-window comparison) or is a constant copied
from PLAN.md / the checkpoint reports, with its source named next to it. tests/test_findings.py checks the
constants against the CSVs where both exist, so a page number cannot drift away from the reports.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

VERSIONS = ["v0-poisson", "v1-working", "v2-shots"]

# Test window 2023/24 to 2025/26, 1,140 matches. Baselines: PLAN.md "Checkpoint 1 results".
BASELINES = [
    ("Uniform 1/3", 1.0986),
    ("Training class frequencies", 1.0728),
]
# Rows that are not in a CSV: (label, test log loss, validation log loss) from PLAN.md "Results" table.
FROM_PLAN = {
    "cd + promoted, joint": (0.9789, 0.9746),
    "working + team home adv": (0.9805, 0.9736),
}
BOOK_TEST_LL = 0.9650  # Avg odds, margin removed, same 1,140 matches (PLAN.md, model_comparison.md)

# Shots model minus working model, change in log loss (negative = shots better). PLAN.md "Results".
# (window, xi_s=0.6 change, xi_s=1.0 change, +- for 0.6, +- for 1.0)
SHOTS_WINDOWS = [
    ("Early 2018-21", -0.0042, -0.0034, 0.0030),
    ("Validation 2021-23", 0.0014, 0.0007, 0.0040),
    ("Test 2023-26", -0.0043, -0.0045, 0.0025),
]
SHOTS_POOLED = (-0.003, 0.0017)  # pooled change and its SE, about 1.9 SE (PLAN.md)

DIDNT_WORK = [
    ("Dixon-Coles correction", "Fitted ρ = −0.09 (low scores 0–0 and 1–1 slightly more likely than independence says), but the gain in log loss, 0.0007 ± 0.0008, is inside the noise. Not adopted."),
    ("Team-specific home advantage", "Fitted team deltas were small (sd 0.05) and validation did not improve. Not adopted."),
    ("Closed-doors term", "Needed to describe 2020/21: home advantage was about 0.18 with fans and about 0.03 without. Improves description, but changes log loss by only 0.0001."),
    ("Promoted-team offsets", "Across three windows the effect is +0.0008 (worse), 0.0000 and −0.0036 (better): pooled about zero. Kept as harmless, but unproven."),
]

LIMITS = [
    "No injuries, line-ups or transfers: the model only sees results and shots on target. The bookmakers see all of that.",
    "A team with no earlier match in the data (such as a newly promoted side in its first game) cannot be rated, so that match is skipped.",
    "The test window was looked at many times while choosing models, so its numbers are indicative, not clean. Validation chose; the early window and live data are the independent checks.",
    "Standard errors treat matches as independent, which understates the real uncertainty because a team's matches are related.",
    "Live sample is tiny. About 350 matches are needed to confirm a 0.0165 gap at 2 standard errors; one round is 10.",
    "Bookmaker odds are the average across firms taken from the match table (captured around the match), not a locked pre-kickoff price, so the live benchmark is slightly stricter than a real bet.",
]


def _read(path):
    return pd.read_csv(path) if Path(path).exists() else None


def season_gaps(reports="reports"):
    """Per-season paired gap to B365 (model log loss minus bookmaker log loss) for each model version."""
    out = []
    frames = {v: _read(f"{reports}/gap_by_season_{v}.csv") for v in VERSIONS}
    base = frames["v0-poisson"]
    if base is None:
        return out
    for i, row in base.iterrows():
        rec = {"season": row["season"], "n": int(row["n"]), "goals_pm": float(row["goals_pm"])}
        for v, f in frames.items():
            if f is None:
                continue
            r = f[f["season"] == row["season"]]
            if len(r) and not math.isnan(r.iloc[0]["gap_B365"]):
                rec[v] = [round(float(r.iloc[0]["gap_B365"]), 4), round(float(r.iloc[0]["se_B365"]), 4)]
        out.append(rec)
    return out


def ladder(reports="reports"):
    """Rows (label, test log loss, validation log loss or None, gap to Avg odds, SE) ordered worst to best."""
    rows = []
    for lab, ll in BASELINES:
        rows.append({"label": lab, "ll": ll, "kind": "baseline"})
    m = _read(f"{reports}/models/test_comparison.csv")
    s = _read(f"{reports}/models_shots/test_comparison.csv")
    names = {
        "B: Poisson": "Poisson goals model",
        "B + Dixon-Coles": "+ Dixon-Coles",
        "B + closed doors": "+ closed doors",
        "working: cd + promoted (2-stage)": "+ promoted teams (v1)",
        "working + shots (xi_s=1.0)": "+ shots on target (v2)",
    }
    for frame in (m, s):
        if frame is None:
            continue
        for _, r in frame.iterrows():
            if r["model"] in names:
                rows.append({"label": names[r["model"]], "ll": round(float(r["log_loss"]), 4), "kind": "model",
                             "gap": round(float(r["dLL_vs_Avg"]), 4), "se": round(float(r["se_Avg"]), 4)})
    rows.append({"label": "Bookmakers (average odds)", "ll": BOOK_TEST_LL, "kind": "book"})
    seen, uniq = set(), []
    for r in rows:
        if r["label"] not in seen:
            seen.add(r["label"])
            uniq.append(r)
    order = {"baseline": 0, "model": 1, "book": 2}
    uniq.sort(key=lambda r: (order[r["kind"]], -r["ll"]))
    return uniq


def build(reports="reports"):
    return {
        "seasons": season_gaps(reports),
        "ladder": ladder(reports),
        "shots": {"windows": [{"name": n, "a": a, "b": b, "se": se} for n, a, b, se in SHOTS_WINDOWS],
                  "pooled": list(SHOTS_POOLED)},
        "didnt": [{"name": n, "text": t} for n, t in DIDNT_WORK],
        "limits": LIMITS,
        "book_test_ll": BOOK_TEST_LL,
    }

"""Run the walk-forward evaluation of the Poisson model against the bookmaker baselines.

    python -m src.run_walkforward                 # quick grid
    python -m src.run_walkforward --grid full     # larger grid (slower)

Protocol (no test leakage):
  1. Choose the decay rate xi and ridge strength kappa by walk-forward log loss on a
     VALIDATION window (2021/22 and 2022/23).
  2. With those two values fixed, score the TEST window (2023/24 to 2025/26) once.
  3. Compare with uniform, class-frequency and bookmaker baselines on the same matches.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .walkforward import fit_before, grid_search, load_matches, team_strengths, walk_forward

VAL_START, VAL_END = "2021-07-01", "2023-07-01"  # 2021/22 and 2022/23
TEST_START, TEST_END = "2023-07-01", "2026-07-01"  # 2023/24 to 2025/26

GRIDS = {
    # xi is the time-decay rate per year (half-life = ln 2 / xi); kappa is the ridge strength
    "quick": {"xi": [0.0, 0.2, 0.5, 1.0], "kappa": [1.0, 5.0, 20.0]},
    "full": {"xi": [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], "kappa": [0.5, 2.0, 5.0, 10.0, 20.0, 40.0]},
    # for when the quick grid's best xi sits on its upper edge: probes much faster decay
    "wide": {"xi": [0.5, 1.0, 1.5, 2.0, 3.0], "kappa": [2.0, 5.0, 10.0]},
}

ODDS_SOURCES = {
    "B365": ["B365H", "B365D", "B365A"],
    "Avg": ["AvgH", "AvgD", "AvgA"],
    "PS": ["PSH", "PSD", "PSA"],
}


def md_table(df, formats=None):
    """Plain markdown table (no extra dependencies)."""
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, formats[c]) if c in formats and pd.notna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate_sources(test, model_probs, train_freqs):
    """Score the model and baselines on `test` rows. Returns (full-window, common-subset) tables."""
    y_all = test["y"].values

    def one(label, probs, y):
        return {"predictor": label, "matches": len(y), **M.scores(probs, y)}

    def table(mask_all, rows_label):
        rows = []
        y = y_all[mask_all]
        rows.append(one("Poisson model", model_probs[mask_all], y))
        rows.append(one("Uniform 1/3", np.full((len(y), 3), 1 / 3), y))
        rows.append(one("Training class frequencies", np.tile(train_freqs, (len(y), 1)), y))
        return rows

    # full window: every predictor on every match it has data for
    rows_full = table(np.ones(len(test), dtype=bool), "all")
    for name, cols in ODDS_SOURCES.items():
        if not all(c in test.columns for c in cols):
            continue
        mask = test[cols].notna().all(axis=1).values
        if mask.sum() == 0:
            continue
        odds = test.loc[mask, cols].values.astype(float)
        rows_full.append(one(f"{name} (margin removed)", M.implied_probs(odds), y_all[mask]))
        rows_full[-1]["model_on_same_matches"] = M.log_loss(model_probs[mask], y_all[mask])

    # common subset: matches where every odds source is present
    cols_all = [c for cols in ODDS_SOURCES.values() for c in cols]
    common = None
    if all(c in test.columns for c in cols_all):
        mask = test[cols_all].notna().all(axis=1).values
        if mask.sum() > 0:
            rows = table(mask, "common")
            for name, cols in ODDS_SOURCES.items():
                odds = test.loc[mask, cols].values.astype(float)
                rows.append(one(f"{name} (margin removed)", M.implied_probs(odds), y_all[mask]))
            common = pd.DataFrame(rows)
    return pd.DataFrame(rows_full), common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/matches.csv")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="quick")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"{data_path} not found. Create it first with: python src/data.py")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, teams = load_matches(data_path)
    T = len(teams)
    print(f"Loaded {len(df)} matches, {T} teams, {df['Date'].min().date()} to {df['Date'].max().date()}")

    grid = GRIDS[args.grid]

    def progress(i, total, row):
        remaining = (total - i) * row["seconds"]
        print(
            f"  [{i:>2}/{total}] xi={row['xi']:<4} kappa={row['kappa']:<5} "
            f"validation log loss {row['log_loss']:.4f}  ({row['seconds']:.1f}s, ~{remaining/60:.1f} min left)",
            flush=True,
        )

    # ---- 1. choose hyperparameters on the validation window -----------------------------
    print(f"\nValidation window {VAL_START} to {VAL_END}, grid '{args.grid}':")
    val = grid_search(df, T, VAL_START, VAL_END, grid["xi"], grid["kappa"], on_result=progress)
    best = val.sort_values("log_loss").iloc[0]
    xi, kappa = float(best["xi"]), float(best["kappa"])
    half_life = np.log(2) / xi if xi > 0 else float("inf")
    print(f"\nChosen on validation: xi={xi} (half-life {half_life:.1f} years), kappa={kappa}, log loss {best['log_loss']:.4f}")
    edge = (xi in (min(grid["xi"]), max(grid["xi"])) and len(grid["xi"]) > 2) or kappa in (
        min(grid["kappa"]),
        max(grid["kappa"]),
    )
    if edge:
        print("Note: the best value sits on the edge of the grid; a wider grid may do better.")

    # ---- 2. score the test window once --------------------------------------------------
    print(f"\nTest window {TEST_START} to {TEST_END} (scored once, hyperparameters fixed)...", flush=True)
    t0 = time.time()
    wf = walk_forward(df, T, TEST_START, TEST_END, xi, kappa)
    test = df.loc[wf.index]
    pre_test = df[df["Date"] < pd.Timestamp(TEST_START)]
    freqs = M.class_frequencies(pre_test["y"].values)
    full, common = evaluate_sources(test, wf.values, freqs)
    print(f"  done in {time.time() - t0:.1f}s")

    fmt = {"log_loss": ".4f", "brier": ".4f", "accuracy": ".1%", "model_on_same_matches": ".4f"}
    print("\nTest window, each predictor on the matches it has data for:")
    print(full.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if common is not None:
        print("\nTest window, only matches where every odds source is present (fair comparison):")
        print(common.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 3. fitted strengths on all data ------------------------------------------------
    last = df["Date"].max() + pd.Timedelta(days=1)
    final = fit_before(df, last, T, xi, kappa)
    iota, eta, _, _ = final.unpack(T)
    strengths = team_strengths(final.theta, teams, T)
    recent = set(df.loc[df["Date"] >= last - pd.Timedelta(days=400), ["HomeTeam", "AwayTeam"]].values.ravel())
    strengths = strengths[strengths["team"].isin(recent)].round(3)
    print(f"\nFit on all {len(df)} matches: league level iota={iota:.3f}, home advantage eta={eta:.3f}")
    print(f"  -> average away goals {np.exp(iota):.2f}, average home goals {np.exp(iota + eta):.2f}")
    print("\nTop 5 attacks:")
    print(strengths.sort_values("attack", ascending=False).head(5).to_string(index=False))
    print("\nTop 5 defences:")
    print(strengths.sort_values("defence", ascending=False).head(5).to_string(index=False))

    # ---- 4. save ------------------------------------------------------------------------
    val.to_csv(out_dir / "validation_grid.csv", index=False)
    pred = test[["Date", "HomeTeam", "AwayTeam", "FTR"]].join(wf)
    pred.to_csv(out_dir / "test_predictions.csv", index=False)
    lines = [
        "# Walk-forward results: independent Poisson model",
        "",
        f"Validation window: {VAL_START} to {VAL_END}. Test window: {TEST_START} to {TEST_END}.",
        f"Chosen on validation: xi = {xi} per year (half-life {half_life:.1f} years), kappa = {kappa}.",
        "",
        "## Test window: each predictor on the matches it has data for",
        md_table(full, fmt),
        "",
    ]
    if common is not None:
        lines += ["## Test window: matches where every odds source is present", md_table(common, fmt), ""]
    lines += [
        "## Validation grid (log loss)",
        md_table(val[["xi", "kappa", "log_loss", "brier", "accuracy"]], {"log_loss": ".4f", "brier": ".4f", "accuracy": ".1%"}),
        "",
        "## Fitted strengths on all data (teams active in the last 400 days)",
        f"iota = {iota:.3f}, eta = {eta:.3f}",
        "",
        md_table(strengths.sort_values("attack", ascending=False)),
        "",
    ]
    (out_dir / "walkforward_results.md").write_text("\n".join(lines))
    print(f"\nSaved results to {out_dir}/ (walkforward_results.md, validation_grid.csv, test_predictions.csv)")


if __name__ == "__main__":
    main()

"""Ablation: does the Dixon-Coles correction and/or a closed-doors home advantage help?

    python -m src.compare_models

Four models, each tuned separately on the VALIDATION window and then scored once on the TEST window:
    B        independent Poisson
    B + cd   + closed-doors shift gamma on the home advantage
    B + dc   + Dixon-Coles low-score correction rho
    C        both
Differences are reported as paired per-match log-loss differences with a standard error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P
from .run_walkforward import ODDS_SOURCES, TEST_END, TEST_START, VAL_END, VAL_START, md_table
from .walkforward import BASE, ModelSpec, fit_before, grid_search, load_matches, walk_forward

SPECS = [
    BASE,
    ModelSpec("B + closed doors", closed_doors=True),
    ModelSpec("B + Dixon-Coles", dc=True),
    ModelSpec("C: both", closed_doors=True, dc=True),
]
GRID = {"xi": [0.75, 1.0, 1.5, 2.0], "kappa": [2.0, 5.0]}


def per_match_loss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def paired(a, b):
    """Mean and standard error of the per-match difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--out-dir", default="reports/models")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    T = len(teams)
    print(f"Loaded {len(df)} matches; {int(df['closed_doors'].sum())} flagged as played without fans")

    chosen, test_probs, val_tables = {}, {}, []
    for spec in SPECS:
        print(f"\n== {spec.name} ==  validation {VAL_START} to {VAL_END}")
        val = grid_search(
            df, T, VAL_START, VAL_END, GRID["xi"], GRID["kappa"], spec=spec,
            on_result=lambda i, n, r: print(f"  [{i}/{n}] xi={r['xi']} kappa={r['kappa']} LL {r['log_loss']:.4f}", flush=True),
        )
        best = val.sort_values("log_loss").iloc[0]
        chosen[spec.name] = (float(best["xi"]), float(best["kappa"]), float(best["log_loss"]))
        val_tables.append(val.assign(model=spec.name))
        print(f"  chosen xi={best['xi']} kappa={best['kappa']} (validation LL {best['log_loss']:.4f})")
        test_probs[spec.name] = walk_forward(df, T, TEST_START, TEST_END, best["xi"], best["kappa"], spec=spec)

    # ---- test window ----------------------------------------------------------------------
    idx = test_probs[BASE.name].index
    test = df.loc[idx]
    y = test["y"].values
    base_loss = per_match_loss(test_probs[BASE.name].values, y)

    avg_cols = ODDS_SOURCES["Avg"]
    have_odds = all(c in test.columns for c in avg_cols)
    has_avg = test[avg_cols].notna().all(axis=1).values if have_odds else np.zeros(len(test), dtype=bool)
    avg_loss = np.full(len(test), np.nan)
    if has_avg.any():
        avg_loss[has_avg] = per_match_loss(M.implied_probs(test.loc[has_avg, avg_cols].values.astype(float)), y[has_avg])

    rows = []
    for spec in SPECS:
        p = test_probs[spec.name].values
        loss = per_match_loss(p, y)
        d_b, se_b = paired(loss, base_loss)
        d_o, se_o = paired(loss[has_avg], avg_loss[has_avg]) if has_avg.sum() > 1 else (np.nan, np.nan)
        xi, kappa, _ = chosen[spec.name]
        rows.append({
            "model": spec.name, "xi": xi, "kappa": kappa, "matches": len(y), **M.scores(p, y),
            "dLL_vs_B": d_b, "se_B": se_b, "dLL_vs_Avg": d_o, "se_Avg": se_o,
        })
    if has_avg.any():
        sc = M.scores(M.implied_probs(test.loc[has_avg, avg_cols].values.astype(float)), y[has_avg])
        rows.append({"model": "Avg odds (margin removed)", "matches": int(has_avg.sum()), **sc})
    table = pd.DataFrame(rows)
    print(f"\nTest window {TEST_START} to {TEST_END} (negative dLL = better than the reference):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- fitted extras on all data --------------------------------------------------------
    last = df["Date"].max() + pd.Timedelta(days=1)
    fit_lines = []
    for spec in (SPECS[1], SPECS[3]):
        xi, kappa, _ = chosen[spec.name]
        res = fit_before(df, last, T, xi, kappa, spec=spec)
        _, eta, _, _ = res.unpack(T)
        gamma, rho = res.unpack_extra(T, 1, spec.dc)
        fit_lines.append(
            f"{spec.name}: eta (with fans) = {eta:.3f}; gamma = {gamma[0]:.3f}; "
            f"eta + gamma (no fans) = {eta + gamma[0]:.3f}; rho = {rho:.3f}"
        )
    spec_dc = SPECS[2]
    xi, kappa, _ = chosen[spec_dc.name]
    res = fit_before(df, last, T, xi, kappa, spec=spec_dc)
    fit_lines.append(f"{spec_dc.name}: rho = {res.unpack_extra(T, 0, True)[1]:.3f}")
    print("\nFitted on all data:")
    for line in fit_lines:
        print("  " + line)

    # ---- save -----------------------------------------------------------------------------
    pd.concat(val_tables).to_csv(out / "validation_grids.csv", index=False)
    table.to_csv(out / "test_comparison.csv", index=False)
    (out / "model_comparison.md").write_text(
        "# Model comparison (test window)\n\n" + md_table(table, {c: ".4f" for c in table.columns if c not in ("model", "matches")})
        + "\n\n## Fitted extras\n\n" + "\n".join(f"- {l}" for l in fit_lines) + "\n"
    )
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()

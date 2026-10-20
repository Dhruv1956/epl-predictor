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
from .walkforward import BASE, WORKING, ModelSpec, fit_before, grid_search, load_matches, walk_forward

SUITES = {
    # first spec in each suite is the reference for the paired differences
    "ablation": [
        BASE,
        ModelSpec("B + closed doors", closed_doors=True),
        ModelSpec("B + Dixon-Coles", dc=True),
        ModelSpec("C: both", closed_doors=True, dc=True),
    ],
    "promo": [
        ModelSpec("B + closed doors", closed_doors=True),
        ModelSpec("B + cd + promoted", closed_doors=True, promoted=True),
        ModelSpec("C + promoted", closed_doors=True, dc=True, promoted=True),
    ],
    # does it help to estimate the promoted offsets with slower decay than the team strengths?
    "promo2": [
        ModelSpec("B + closed doors", closed_doors=True),
        ModelSpec("promoted, joint", closed_doors=True, promoted=True),
        ModelSpec("promoted, offsets xi=0", closed_doors=True, promoted=True, promo_xi=0.0),
        ModelSpec("promoted, offsets xi=0.3", closed_doors=True, promoted=True, promo_xi=0.3),
    ],
    # does each team have its own home advantage? (kappa_home: larger = shrunk harder towards one common value)
    "homeadv": [
        WORKING,
        ModelSpec("working + team home adv (kappa_h=20)", closed_doors=True, promoted=True, promo_xi=0.0, kappa_home=20.0),
        ModelSpec("working + team home adv (kappa_h=60)", closed_doors=True, promoted=True, promo_xi=0.0, kappa_home=60.0),
        ModelSpec("working + team home adv (kappa_h=5)", closed_doors=True, promoted=True, promo_xi=0.0, kappa_home=5.0),
    ],
    # do shots on target (from earlier matches) improve the goal model? shots_xi = how fast shot evidence decays
    "shots": [
        WORKING,
        ModelSpec("working + shots (xi_s=0.3)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=0.3),
        ModelSpec("working + shots (xi_s=0.6)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=0.6),
        ModelSpec("working + shots (xi_s=1.0)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=1.0),
    ],
}
GRID = {"xi": [0.75, 1.0, 1.5, 2.0], "kappa": [2.0, 5.0]}


def per_match_loss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def paired(a, b):
    """Mean and standard error of the per-match difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def subset_table(frame, probs, specs, ref, label):
    """Paired log-loss differences vs the reference, split by 'involves a newly promoted team'."""
    y = frame["y"].values
    newcomer = ((frame["h_new"] + frame["a_new"]) > 0).values
    ref_loss = per_match_loss(probs[ref.name].values, y)
    rows = []
    allm = np.ones(len(y), dtype=bool)
    for name, mask in (("all matches", allm), ("involving a newly promoted team", newcomer), ("all other matches", ~newcomer)):
        if mask.sum() < 2:
            continue  # nothing to compare (e.g. no newly promoted teams in this data)
        for spec in specs[1:]:
            loss = per_match_loss(probs[spec.name].values, y)
            d, se = paired(loss[mask], ref_loss[mask])
            rows.append({"window": label, "subset": name, "matches": int(mask.sum()), "model": spec.name,
                         "ref_log_loss": ref_loss[mask].mean(), "log_loss": loss[mask].mean(),
                         "dLL_vs_ref": d, "se": se})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--suite", choices=sorted(SUITES), default="ablation")
    ap.add_argument("--out-dir", default=None, help="default: reports/models (ablation) or reports/models_<suite>")
    args = ap.parse_args(argv)
    SPECS = SUITES[args.suite]
    REF = SPECS[0]

    out = Path(args.out_dir or ("reports/models" if args.suite == "ablation" else f"reports/models_{args.suite}"))
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    if any(sp.shots for sp in SPECS) and "HST" not in df.columns:
        raise SystemExit("Shots data not found. Run `python -m src.add_shots` first (it writes data/processed/shots.csv).")
    if any(sp.shots for sp in SPECS):
        print(f"Shots-on-target data present for {df['HST'].notna().mean():.1%} of matches")
    T = len(teams)
    n_new = int(((df["h_new"] + df["a_new"]) > 0).sum())
    print(f"Loaded {len(df)} matches; {int(df['closed_doors'].sum())} flagged as played without fans; "
          f"{n_new} involve a newly promoted team")

    chosen, test_probs, val_probs, val_tables = {}, {}, {}, []
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
        val_probs[spec.name] = walk_forward(df, T, VAL_START, VAL_END, best["xi"], best["kappa"], spec=spec)
        test_probs[spec.name] = walk_forward(df, T, TEST_START, TEST_END, best["xi"], best["kappa"], spec=spec)

    # ---- test window ----------------------------------------------------------------------
    idx = test_probs[REF.name].index
    test = df.loc[idx]
    y = test["y"].values
    base_loss = per_match_loss(test_probs[REF.name].values, y)
    newcomer = ((test["h_new"] + test["a_new"]) > 0).values

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
            "dLL_vs_ref": d_b, "se_ref": se_b, "dLL_vs_Avg": d_o, "se_Avg": se_o,
        })
    if has_avg.any():
        sc = M.scores(M.implied_probs(test.loc[has_avg, avg_cols].values.astype(float)), y[has_avg])
        rows.append({"model": "Avg odds (margin removed)", "matches": int(has_avg.sum()), **sc})
    table = pd.DataFrame(rows)
    print(f"\nTest window {TEST_START} to {TEST_END} (reference model: {REF.name}; negative dLL = better):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- where does the difference come from? matches with / without a newcomer ------------
    sub = pd.concat([
        subset_table(df.loc[val_probs[REF.name].index], val_probs, SPECS, REF, "validation"),
        subset_table(test, test_probs, SPECS, REF, "test"),
    ])
    print("\nSplit by matches involving a newcomer (each model uses its own chosen xi, kappa):")
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- fitted extras on all data --------------------------------------------------------
    last = df["Date"].max() + pd.Timedelta(days=1)
    fit_lines = []
    for spec in SPECS:
        names = spec.covariate_names()
        if not names and not spec.dc and spec.kappa_home is None:
            continue
        xi, kappa, _ = chosen[spec.name]
        res = fit_before(df, last, T, xi, kappa, spec=spec)
        _, eta, _, _ = res.unpack(T)
        ha = spec.kappa_home is not None
        gamma, rho = res.unpack_extra(T, len(names), spec.dc, ha)
        parts = [f"eta = {eta:.3f}"] + [f"{n} = {g:.3f}" for n, g in zip(names, gamma)]
        if spec.dc:
            parts.append(f"rho = {rho:.3f}")
        if ha:
            delta = pd.Series(res.unpack_delta(T, len(names)), index=teams)
            active = set(df.loc[df["Date"] >= last - pd.Timedelta(days=400), ["HomeTeam", "AwayTeam"]].values.ravel())
            d = delta[delta.index.isin(active)].sort_values()
            parts.append(f"delta sd = {d.std():.3f}; lowest {', '.join(f'{t} {v:+.2f}' for t, v in d.head(3).items())}; "
                         f"highest {', '.join(f'{t} {v:+.2f}' for t, v in d.tail(3).items())}")
        fit_lines.append(f"{spec.name}: " + "; ".join(parts))
    print("\nFitted on all data (coefficients are log-rate shifts; negative = weaker):")
    for line in fit_lines:
        print("  " + line)

    # ---- save -----------------------------------------------------------------------------
    pd.concat(val_tables).to_csv(out / "validation_grids.csv", index=False)
    table.to_csv(out / "test_comparison.csv", index=False)
    sub.to_csv(out / "test_by_subset.csv", index=False)
    (out / "model_comparison.md").write_text(
        "# Model comparison (test window)\n\n" + md_table(table, {c: ".4f" for c in table.columns if c not in ("model", "matches")})
        + "\n\n## Split by subset\n\n" + md_table(sub, {c: ".4f" for c in ("ref_log_loss", "log_loss", "dLL_vs_ref", "se")})
        + "\n\n## Fitted extras\n\n" + "\n".join(f"- {l}" for l in fit_lines) + "\n"
    )
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()

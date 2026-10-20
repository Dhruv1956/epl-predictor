"""Independent confirmation on an EARLY window that no model choice was based on.

    python -m src.early_check

Every model decision so far used the validation window (2021/22-2022/23); the test window (2023/24-2025/26)
has been looked at for many variants. This script scores the candidates on 2018/19-2020/21 instead, a window
that was never used to choose a model or to fit hyperparameters. The decay rate and ridge strength of each
candidate are the ones already chosen on the validation window (no re-tuning here). Training history is
shorter at the start of this window (3 seasons), which is a handicap shared by all candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .compare_models import SUITES, paired, per_match_loss, subset_table
from .run_walkforward import ODDS_SOURCES, md_table
from .walkforward import BASE, WORKING, ModelSpec, load_matches, walk_forward

# (spec, xi, kappa): xi and kappa as chosen on the validation window in earlier runs
CANDIDATES = [
    (BASE, 1.0, 5.0),
    (ModelSpec("B + closed doors", closed_doors=True), 1.0, 5.0),
    (ModelSpec("C: cd + Dixon-Coles", closed_doors=True, dc=True), 1.0, 5.0),
    (ModelSpec("cd + promoted, joint", closed_doors=True, promoted=True), 0.75, 5.0),
    (WORKING, 1.0, 5.0),
    (ModelSpec("working + team home adv (kappa_h=20)", closed_doors=True, promoted=True, promo_xi=0.0, kappa_home=20.0), 1.0, 5.0),
    (ModelSpec("working + shots (xi_s=0.6)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=0.6), 1.5, 5.0),
    (ModelSpec("working + shots (xi_s=1.0)", closed_doors=True, promoted=True, promo_xi=0.0, shots=True, shots_xi=1.0), 1.5, 5.0),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--start", default="2018-07-01")
    ap.add_argument("--end", default="2021-07-01")
    ap.add_argument("--out-dir", default="reports/early_check")
    args = ap.parse_args(argv)

    df, teams = load_matches(args.data)
    if "HST" not in df.columns:
        raise SystemExit("Shots data not found. Run `python -m src.add_shots` first.")
    T = len(teams)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    probs = {}
    for spec, xi, kappa in CANDIDATES:
        print(f"{spec.name} (xi={xi}, kappa={kappa}) ...", flush=True)
        probs[spec.name] = walk_forward(df, T, args.start, args.end, xi, kappa, spec=spec)

    ref = WORKING
    frame = df.loc[probs[ref.name].index]
    y = frame["y"].values
    ref_loss = per_match_loss(probs[ref.name].values, y)

    # bookmaker benchmark where odds exist (Avg is missing before 2019/20, so fall back to B365)
    cols = ODDS_SOURCES["B365"]
    has = frame[cols].notna().all(axis=1).values if all(c in frame.columns for c in cols) else np.zeros(len(frame), bool)
    book_loss = np.full(len(frame), np.nan)
    if has.any():
        book_loss[has] = per_match_loss(M.implied_probs(frame.loc[has, cols].values.astype(float)), y[has])

    rows = []
    for spec, xi, kappa in CANDIDATES:
        p = probs[spec.name].values
        loss = per_match_loss(p, y)
        d, se = paired(loss, ref_loss) if spec.name != ref.name else (0.0, 0.0)
        db, seb = paired(loss[has], book_loss[has]) if has.sum() > 1 else (np.nan, np.nan)
        rows.append({"model": spec.name, "xi": xi, "kappa": kappa, "matches": len(y), **M.scores(p, y),
                     "dLL_vs_working": d, "se": se, "dLL_vs_B365": db, "se_B365": seb})
    table = pd.DataFrame(rows)
    print(f"\nEarly window {args.start} to {args.end} (reference: {ref.name}; negative = better than the reference;")
    print(f"bookmaker comparison on the {int(has.sum())} matches with B365 odds):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    specs = [c[0] for c in CANDIDATES]
    ordered = [ref] + [s for s in specs if s.name != ref.name]
    sub = subset_table(frame, probs, ordered, ref, "early")
    print("\nSplit by matches involving a newly promoted team:")
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    table.to_csv(out / "early_check.csv", index=False)
    sub.to_csv(out / "early_check_by_subset.csv", index=False)
    (out / "early_check.md").write_text(
        "# Early-window check\n\n" + md_table(table, {c: ".4f" for c in table.columns if c not in ("model", "matches")})
        + "\n\n## Split by subset\n\n" + md_table(sub, {c: ".4f" for c in ("ref_log_loss", "log_loss", "dLL_vs_ref", "se")}) + "\n"
    )
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()

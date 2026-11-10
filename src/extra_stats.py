"""Do total shots or corners add information on top of the v2-shots model?

v2 already feeds the model team strengths learned from shots ON TARGET (stage 0, earlier matches only, one coefficient c).
Here each extra statistic gets the same treatment: its own stage-0 Poisson fit on earlier matches,
    HS ~ Poisson(exp(i + e + a_h - b_a)),  AS ~ Poisson(exp(i + a_a - b_h))     (HC, AC for corners)
and its attack-minus-defence strength enters the goal rates through its own learned coefficient:
    log lam += c_k (a^k_h - b^k_a),   log mu += c_k (a^k_a - b^k_h).
Same-match values are never used for that match. Variants (all start from v2): + total shots, + corners, + both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from . import poisson as P
from .walkforward import ModelSpec, fit_before, predict_block, week_periods

WINDOWS = {"early": ("2018-07-01", "2021-07-01"), "validation": ("2021-07-01", "2023-07-01"), "test": ("2023-07-01", "2026-07-01")}
EXTRAS = {"+ total shots": (("HS", "AS", 1.0, 5.0),), "+ corners": (("HC", "AC", 1.0, 5.0),),
          "+ shots and corners": (("HS", "AS", 1.0, 5.0), ("HC", "AC", 1.0, 5.0))}


def with_extras(spec, extras):
    from dataclasses import replace
    return replace(spec, extra_stats=tuple(extras))


def walk_forward_coef(df, T, start, end, xi, kappa, spec):
    """Like walkforward.walk_forward but also returns each block's fitted covariate coefficients (DataFrame)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]
    names = spec.covariate_names()
    theta, idx, probs, coefs = None, [], [], []
    for _, block in window.groupby(week_periods(window["Date"]), sort=True):
        res = fit_before(df, block["Date"].min(), T, xi, kappa, theta0=theta, spec=spec)
        theta = res.theta
        probs.append(predict_block(theta, block, T, spec, res.aux))
        idx.append(block.index.values)
        coefs.append(theta[2 + 2 * T: 2 + 2 * T + len(names)])
    out = pd.DataFrame(np.vstack(probs), index=np.concatenate(idx), columns=["pH", "pD", "pA"]).sort_index()
    return out, pd.DataFrame(coefs, columns=names)


def main(argv=None):
    import argparse
    from pathlib import Path
    from .live import MODELS
    from .walkforward import load_matches
    ap = argparse.ArgumentParser(description="Total shots and corners on top of v2-shots.")
    ap.add_argument("--data", default="data/processed/matches.csv")
    ap.add_argument("--out", default="reports/extra_stats")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, teams = load_matches(args.data)
    for c in ("HS", "AS", "HC", "AC"):
        if c not in df.columns or df[c].isna().all():
            raise SystemExit(f"column {c} is missing: run python -m src.add_shots")
    T = len(teams)
    cfg = MODELS["v2-shots"]
    variants = {"v2-shots": cfg["spec"], **{k: with_extras(cfg["spec"], v) for k, v in EXTRAS.items()}}
    rows, coefrows = [], []
    ctrl = {}
    for w, (s, e) in WINDOWS.items():
        for name, spec in variants.items():
            wf, co = walk_forward_coef(df, T, s, e, cfg["xi"], cfg["kappa"], spec)
            y = df.loc[wf.index, "y"].values
            ll = -np.log(np.clip(wf.values[np.arange(len(y)), y], 1e-15, 1))
            if name == "v2-shots":
                ctrl[w] = ll
            d = ll - ctrl[w]
            rows.append({"window": w, "variant": name, "n": len(y), "log_loss": ll.mean(), "brier": M.brier_score(wf.values, y),
                         "delta_vs_v2": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)) if name != "v2-shots" else 0.0})
            coefrows.append({"window": w, "variant": name, **co.mean().to_dict()})
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "extra_stats_results.csv", index=False)
    pd.DataFrame(coefrows).to_csv(out / "extra_stats_coefficients.csv", index=False)
    print(pd.DataFrame(coefrows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

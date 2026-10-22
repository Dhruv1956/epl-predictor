"""Frozen per-match explanations: why the shots model (v2) predicted what it did.

    python -m src.explain --fixtures data/live/fixtures_manual.csv --verify   # backfill + check against the log

For each fixture this records, at prediction time and from the very same fit as the logged predictions:
  * the expected-goals build-up as additive terms on the log scale (base, home advantage, shots signal,
    promoted-team offsets, "results vs chances" = the ridge-shrunk goal-strength deviations);
  * each team's rank among the current season's clubs for shots on target created / allowed and for the
    combined attack / defence strength (goal deviation + c * shot strength);
  * each team's last six results (with shots on target) and the last five meetings.
Records are appended to live/explanations.jsonl and never overwritten, like the prediction log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import poisson as P
from .live import MODELS, _prepare, read_fixtures
from .walkforward import _covariates, fit_before

MODEL = "v2-shots"
DEFAULT_PATH = "live/explanations.jsonl"
KEY = ("match_date", "home", "away")


def _rank(values, teams, team):
    """Rank (1 = best, i.e. largest value) of `team` among `teams`."""
    ordered = sorted(teams, key=lambda t: -values[t])
    return ordered.index(team) + 1


def recent_matches(played, team, n=6):
    m = played[(played["HomeTeam"] == team) | (played["AwayTeam"] == team)].sort_values("Date").tail(n)
    out = []
    for _, r in m.iterrows():
        home = r["HomeTeam"] == team
        gf, ga = (r["FTHG"], r["FTAG"]) if home else (r["FTAG"], r["FTHG"])
        sf, sa = ("HST", "AST") if home else ("AST", "HST")
        out.append({
            "date": r["Date"].date().isoformat(), "opp": r["AwayTeam"] if home else r["HomeTeam"], "home": bool(home),
            "gf": int(gf), "ga": int(ga), "res": "W" if gf > ga else "D" if gf == ga else "L",
            "sot_for": int(r[sf]) if sf in r.index and pd.notna(r[sf]) else None,
            "sot_against": int(r[sa]) if sa in r.index and pd.notna(r[sa]) else None,
        })
    return out


def head_to_head(played, home, away, n=5):
    m = played[((played["HomeTeam"] == home) & (played["AwayTeam"] == away))
               | ((played["HomeTeam"] == away) & (played["AwayTeam"] == home))].sort_values("Date").tail(n)
    return [{"date": r["Date"].date().isoformat(), "home": r["HomeTeam"], "away": r["AwayTeam"],
             "hg": int(r["FTHG"]), "ag": int(r["FTAG"])} for _, r in m.iterrows()]


def explain_fixtures(matches_path, fixtures, now=None, model=MODEL, models=MODELS):
    """Return one explanation record per fixture (list of dicts, JSON-serialisable)."""
    prep = _prepare(matches_path, fixtures, models)
    T, teams, played, combined, block, ref = (prep[k] for k in ("T", "teams", "played", "combined", "block", "ref"))
    cfg = models[model]
    spec = cfg["spec"]
    res = fit_before(combined, ref, T, cfg["xi"], cfg["kappa"], spec=spec)
    aux = getattr(res, "aux", None)
    iota, eta, alpha, beta = res.unpack(T)
    names = spec.covariate_names()
    gamma, _ = res.unpack_extra(T, len(names), spec.dc, spec.kappa_home is not None)
    gmap = dict(zip(names, gamma))
    X, Xm = _covariates(block, spec, aux)
    h, a = block["h"].values, block["a"].values
    lam, mu = P.rates(res.theta, h, a, T, X, Xm, spec.kappa_home is not None)
    c = gmap.get("shots_strength", 0.0)

    season = played["Season"].iloc[-1] if "Season" in played.columns else None
    cur = played[played["Season"] == season] if season is not None else played
    clubs = sorted(set(cur["HomeTeam"]) | set(cur["AwayTeam"]) | set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    att_goal = {t: alpha[idx[t]] for t in clubs}
    def_goal = {t: beta[idx[t]] for t in clubs}
    if aux is not None:
        sa = {t: aux["a"][idx[t]] for t in clubs}
        sb = {t: aux["b"][idx[t]] for t in clubs}
        att = {t: att_goal[t] + c * sa[t] for t in clubs}
        dfn = {t: def_goal[t] + c * sb[t] for t in clubs}
    else:
        sa = sb = None
        att, dfn = att_goal, def_goal

    recs = []
    for i, (_, f) in enumerate(fixtures.reset_index(drop=True).iterrows()):
        def terms(Xmat):
            t = {nm: float(Xmat[i, k] * gamma[k]) for k, nm in enumerate(names)} if Xmat is not None else {}
            return {
                "shots": t.get("shots_strength", 0.0),
                "promoted": t.get("promoted_attack", 0.0) + t.get("promoted_defence", 0.0),
                "closed_doors": t.get("closed_doors", 0.0),
            }
        hn, an = f["HomeTeam"], f["AwayTeam"]
        tl, tm = terms(X), terms(Xm)
        tl.update(base=float(iota), home=float(eta), adj=float(alpha[h[i]] - beta[a[i]]))
        tm.update(base=float(iota), home=0.0, adj=float(alpha[a[i]] - beta[h[i]]))
        for tt, rate in ((tl, lam[i]), (tm, mu[i])):
            assert abs(sum(tt.values()) - np.log(rate)) < 1e-8, "explanation terms do not add up to the fitted rate"
        rec = {
            "match_date": f["Date"].date().isoformat(), "home": hn, "away": an,
            "kickoff": "" if pd.isna(f["kickoff"]) else f["kickoff"].isoformat(timespec="minutes"),
            "model": model, "ref": ref.date().isoformat(), "n_train": int((combined["Date"] < ref).sum()),
            "lam": float(lam[i]), "mu": float(mu[i]), "terms_lam": tl, "terms_mu": tm,
            "c": float(c), "n_clubs": len(clubs), "season": season,
            "ranks": {
                side: {
                    "att": _rank(att, clubs, t), "def": _rank(dfn, clubs, t),
                    **({"shot_att": _rank(sa, clubs, t), "shot_def": _rank(sb, clubs, t)} if sa is not None else {}),
                } for side, t in (("home", hn), ("away", an))
            },
            "form": {"home": recent_matches(played, hn), "away": recent_matches(played, an)},
            "h2h": head_to_head(played, hn, an),
            "newcomer": {"home": bool(block["h_new"].values[i]), "away": bool(block["a_new"].values[i])},
            "goals_per_match": float((cur["FTHG"] + cur["FTAG"]).mean()) if len(cur) else None,
        }
        recs.append(rec)
    return recs


def append_explanations(records, path=DEFAULT_PATH):
    """Append records whose (match_date, home, away) is not in the file. Returns (n_added, n_already_frozen)."""
    path = Path(path)
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add(tuple(r[k] for k in KEY))
    fresh = [r for r in records if tuple(r[k] for k in KEY) not in seen]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in fresh:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(fresh), len(records) - len(fresh)


def read_explanations(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def max_gap_to_log(records, log_path, model=MODEL):
    """Largest absolute difference between the explanation's lam/mu and the frozen log (should be ~1e-4 from rounding)."""
    log = pd.read_csv(log_path)
    log = log[log["model_version"] == model]
    worst = 0.0
    for r in records:
        row = log[(log["match_date"] == r["match_date"]) & (log["home"] == r["home"]) & (log["away"] == r["away"])]
        if row.empty:
            continue
        worst = max(worst, abs(row["lam"].iloc[0] - r["lam"]), abs(row["mu"].iloc[0] - r["mu"]))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default="data/live/fixtures_manual.csv")
    ap.add_argument("--matches", default="data/processed/matches.csv")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--log", default="live/predictions_log.csv")
    ap.add_argument("--verify", action="store_true", help="check against the frozen prediction log")
    args = ap.parse_args(argv)
    fx = read_fixtures(args.fixtures)
    recs = explain_fixtures(args.matches, fx)
    if args.verify:
        gap = max_gap_to_log(recs, args.log)
        print(f"Largest |lam or mu| difference against the frozen log: {gap:.2e}")
        if gap > 5e-4:
            raise SystemExit("Explanations do not reproduce the frozen predictions: data changed since the freeze? Not writing.")
    added, frozen = append_explanations(recs, args.out)
    print(f"{added} new explanation(s), {frozen} already frozen. File: {args.out}")


if __name__ == "__main__":
    main()

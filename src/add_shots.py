"""Build data/processed/shots.csv: shots and shots on target per match, from the raw season files.

    python -m src.add_shots

These columns are same-match statistics. They are NEVER used to predict the match they belong to
(that would be leakage); the shots model only learns team strength from earlier matches.
`load_matches` merges this file automatically when it sits next to matches.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP = ["Date", "HomeTeam", "AwayTeam", "HS", "AS", "HST", "AST"]


def build_shots(raw_dir):
    frames = []
    for f in sorted(Path(raw_dir).glob("E0_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        missing = [c for c in KEEP if c not in d.columns]
        if missing:
            raise ValueError(f"{f.name} is missing columns {missing}")
        d = d[KEEP].dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
        d["season_file"] = f.stem.replace("E0_", "")
        frames.append(d)
    if not frames:
        raise ValueError(f"no E0_*.csv files found in {raw_dir}")
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(["Date", "HomeTeam", "AwayTeam"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate matches in the raw files")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/shots.csv")
    args = ap.parse_args(argv)
    out = build_shots(args.raw_dir)
    cover = out.groupby("season_file")[["HS", "HST"]].agg(lambda s: s.notna().mean())
    print("Share of matches with shot data, by season file:")
    print(cover.round(3).to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="season_file").to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()

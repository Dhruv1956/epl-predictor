"""Phase 1 data loading: raw football-data.co.uk E0 CSVs -> one tidy table.

Raw files are read-only inputs; the cleaned table goes to data/processed/matches.csv.
Only pre-kickoff information (teams, date, odds) plus the result columns (targets) are kept;
same-match stats such as shots are deliberately dropped so they cannot leak into features.
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "matches.csv"

CORE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
# Bookmaker triples (home, draw, away) we keep when present in a season's file.
ODDS = [f"{b}{o}" for b in ("B365", "Avg", "PS") for o in "HDA"]


def season_label(code: str) -> str:
    """'1516' -> '2015/16'."""
    return f"20{code[:2]}/{code[2:]}"


def parse_date(s: pd.Series) -> pd.Series:
    """Day-first parse of dd/mm/yyyy or dd/mm/yy, element-wise (files mix the two).

    Returns NaT (never guesses) for anything matching neither format.
    """
    s = s.astype("string").str.strip()
    four = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    two = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return four.fillna(two)


def load_season(path: Path) -> pd.DataFrame:
    code = re.fullmatch(r"E0_(\d{4})\.csv", path.name).group(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Raw files can end with blank lines whose cells are all empty.
    df = df.dropna(how="all")
    raw_dates = df["Date"].copy()
    keep = CORE + [c for c in ODDS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = parse_date(raw_dates)
    bad = int(df["Date"].isna().sum())
    if bad:
        print(f"WARNING {path.name}: {bad} unparseable dates")
    df.insert(0, "Season", season_label(code))
    return df


def load_all(write: bool = True) -> pd.DataFrame:
    paths = sorted(RAW_DIR.glob("E0_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no E0_*.csv in {RAW_DIR}")
    df = pd.concat([load_season(p) for p in paths], ignore_index=True)
    # Stable sort: ties on Date keep file order (no kickoff time in older seasons).
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    d = load_all()
    print(f"{len(d)} matches, {d['Date'].min().date()} to {d['Date'].max().date()} -> {OUT_PATH}")

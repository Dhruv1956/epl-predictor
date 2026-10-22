# EPL Match Outcome Predictor

Predicts P(home win), P(draw) and P(away win) for Premier League matches. The models are
independent-Poisson goal models fitted by weighted, ridge-penalised maximum likelihood with
exponential time decay. The log-likelihood and its gradients are derived by hand and implemented
in NumPy/SciPy only (no high-level modelling libraries); the gradients are checked against
numerical gradients in the tests. Predictions are scored with log loss and Brier score against
bookmaker odds with the margin removed.

## Result

The models are about 0.01 to 0.02 log loss behind the bookmakers on the test window. The
shots-on-target stage (v2) closes part of the gap, but the evidence for it is suggestive, not
conclusive.

Test window = 2023/24 to 2025/26 (1,140 matches); validation window = 2021/22 to 2022/23.
"Gap" is log loss minus the bookmakers' Avg odds with the margin removed (0.9650).

| Model | Val log loss | Test log loss | Test gap to Avg odds |
|---|---|---|---|
| v0-poisson (model B: Poisson) | 0.9736 | 0.9852 | +0.0202 |
| v1-working (closed doors + promoted two-stage) | 0.9733 | 0.9815 | +0.0165 |
| v2-shots (v1 + shots, shots_xi=1.0, xi=1.5) | 0.9740 | 0.9770 | +0.0119 |
| Bookmakers (Avg odds, margin removed) | | 0.9650 | |

**Limitations**

- These windows were viewed many times while choosing models and hyperparameters, so the numbers
  are indicative rather than clean out-of-sample results. The live log (below) is the independent
  check.
- v2 versus v1: the change in log loss is -0.0034 +- 0.0030 in the early window (2018-21),
  +0.0007 +- 0.004 in validation, and -0.0045 +- 0.0025 in test. Pooled across the three windows it
  is about -0.003 +- 0.0017, roughly 1.9 standard errors. Suggestive, not conclusive.
- The promoted-team offsets (v1) are unproven.
- No injuries, line-ups or other information beyond results, shots and odds is used.

## Where to look

| Path | Contents |
|---|---|
| `src/` | Model and pipeline code: `poisson.py` (likelihood, gradients, fitting), `walkforward.py` and `metrics.py` (time-based evaluation, log loss, Brier), `data.py` and `audit.py` (loading and data checks), `live.py` (frozen predictions), `explain.py`, `site.py` |
| `tests/` | pytest suite (74 tests), including numerical-gradient checks of the hand-derived gradients and tests on synthetic data with known parameters |
| `live/` | `predictions_log.csv`: append-only log of predictions frozen before kickoff (round 1, 23-25 Oct 2026, is frozen for v0, v1 and v2). `explanations.jsonl`: why each prediction was made |
| `reports/` | `checkpoint1_report.pdf`, `checkpoint2_report.pdf`, plus the tables and CSVs behind them |
| `docs/` | Static weekly website (`index.html`), suitable for GitHub Pages |
| `notebooks/` | Exploration and plots only |
| `PLAN.md`, `CLAUDE.md` | The project plan and working rules |

## Reproducing

**Environment.**

```bash
conda env create -f environment.yml
conda activate epl
```

**Data.** Match data comes from [football-data.co.uk](https://football-data.co.uk/englandm.php)
(`E0_<SEASON>.csv` files in `data/raw/`, which are committed). On the author's network the
`football-data.co.uk` host times out, so PLAN.md documents a `curl --resolve` command that works;
see [Data download in PLAN.md](PLAN.md#data-download). Then build the cleaned table
(`data/processed/` is not committed):

```bash
python src/data.py
```

**Tests.**

```bash
python -m pytest -q
```

**Weekly routine.** Put the upcoming fixtures in `data/live/fixtures.csv`, then:

```bash
python src/data.py                                         # rebuild the match table with new results
python -m src.live predict --fixtures data/live/fixtures.csv   # freeze predictions (append-only)
python -m src.live score                                   # score frozen predictions once results arrive
python -m src.site                                         # regenerate docs/index.html
```

`python -m src.site` also writes `docs/artifact.html`, a fragment for another tool; it is
git-ignored.

## Backtest (rounds already played)

`python -m src.backtest_weeks` replays every round of the current season: for round *k* each model is refitted
using only matches played before that round's first kickoff, predicts the round, and is scored against the real
results and the bookmakers (margin removed, average odds). It writes `backtest/` (a log in the same format as
`live/predictions_log.csv`, per-match explanations, and a note on skipped rounds) and `reports/backtest_season.csv`.
`python -m src.site` then adds a **Backtest** tab next to the live predictions.

This is not a substitute for the live log: the code and settings were fixed before these rounds were examined,
but the matches had already been played when the test was run. Rounds with a team that has no earlier match in
the data (round 1 for a newly promoted side) are skipped. `tests/test_backtest.py` checks that the training set is
exactly the matches before each round, and that changing the results inside a round cannot change its predictions.

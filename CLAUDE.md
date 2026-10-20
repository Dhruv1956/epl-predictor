# EPL Match Outcome Predictor

Predict P(home win / draw / away win) for Premier League matches. See PLAN.md for the full phased plan.

## Rules
- Prefer mathematically deep implementations over library shortcuts: write the Poisson log-likelihood and its gradients by hand in NumPy/SciPy, and verify them against numerical gradients.
- Model code lives in `src/*.py`. Notebooks are for exploration and plots only.
- Split train/test by time, never randomly. Use walk-forward validation.
- Features may only use information available before kickoff. Never use same-match stats (shots, possession) as inputs.
- Evaluate with log loss and Brier score, and compare to bookmaker-odds probabilities.
- Test every model on synthetic data with known parameters before real data.
- Keep a log of bugs and fixes in `reports/bug_log.md` for the final report.
- Explain the maths in detail; the user is a Mathematics and Computing student.

## Data
- Raw CSVs in `data/raw/E0_<SEASON>.csv`, from `https://football-data.co.uk/mmz4281/<SEASON>/E0.csv` (no `www`).
- Never edit raw files; write cleaned data to `data/processed/`.

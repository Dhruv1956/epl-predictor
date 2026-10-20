# EPL Match Outcome Predictor: Project Plan

**Goal:** predict P(home win), P(draw), P(away win) for upcoming Premier League matches, using past performance, current form, and home advantage. Well-calibrated probabilities are the target, not certainty about single matches. Bookmakers get roughly 53-55% of 3-way results right, so matching their log loss is a realistic goal and beating it would be a real finding.

**Working folder (Mac):** `~/epl-predictor/` with `data/raw`, `src`, `notebooks`, `reports`.
**Tools:** VS Code + Jupyter, Claude Code. Model code in `.py` files, notebooks for exploration and plots only.

## Data decisions
- Source: football-data.co.uk, Premier League (E0) CSVs, URL pattern `https://football-data.co.uk/mmz4281/<SEASON>/E0.csv` (no `www`; the `www` host stalled on the user's network).
- Seasons: 2015/16 to current (about 3,500 to 4,000 matches), with exponential time decay. Do not go back past about 2015.
- Reasoning: one season (380 matches, about 42 parameters) is too noisy; very old data is stale. This is the bias-variance trade-off, resolved by time-decay weights e^(-xi * age) with xi chosen by validation.
- Promoted teams have no history: handled by a ridge penalty, which is a Gaussian prior shrinking them to the league average.
- 2020/21 was played behind closed doors: give that season its own home-advantage term.
- `fixtures.csv` is not confirmed to exist at football-data.co.uk. Find a source for upcoming fixtures in Phase 3.

## Phase 1: Plan, setup, Checkpoint 1
1. Outline: problem statement, maths to derive (Poisson goals, likelihood, MLE gradients, Dixon-Coles correction, time decay), evaluation plan.
2. Setup: project folder, Python env, Git repo, notebook, Claude Code, datasets.
3. **Checkpoint 1 report:** problem statement, data audit (missing values, team-name changes, promoted/relegated teams), baselines (always home win, class frequencies), model roadmap.

## Phase 2: Implementation
- Model A: multinomial logistic regression on rolling form/Elo features (baseline).
- Model B: independent Poisson, attack/defence/home-advantage parameters by MLE. Write the log-likelihood and analytic gradients by hand and check them against numerical gradients. NumPy/SciPy only, no high-level model libraries.
- Model C: Dixon-Coles low-score correction plus exponential time decay.
- Time-based train/test split, then walk-forward validation. Features use only information available before kickoff (no leakage).
- Metrics: log loss, Brier score, accuracy; compare against bookmaker-odds baseline.
- Log every bug and fix for the final report.
- Test on synthetic data with known parameters before real data.
- **End of phase:** comparison table of all models.

## Phase 3: Live testing
- Predict the upcoming matchweek, freeze predictions before kickoff, score once results arrive.
- Run the walk-forward backtest alongside, since a live test needs several weeks to mean much.

## Phase 4: Final report
- Maths derivations, results, calibration plot, limitations (no injuries or lineups), next steps (xG data).

## Status
- [x] Plan agreed
- [x] Claude Project created
- [ ] Data downloaded to `~/epl-predictor/data/raw/` (blocked: curl returned empty files; testing the non-www host)
- [ ] Checkpoint 1 report

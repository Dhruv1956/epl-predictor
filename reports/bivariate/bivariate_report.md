# Bivariate Poisson test (31 Oct 2026)

Question: does letting the two sides' goals share a common "tempo" component (X = X1 + Z, Y = Y1 + Z,
Z ~ Poisson(lam3), so Cov(X, Y) = lam3) improve the forecasts? Code: `src/bivariate.py` (likelihood, hand-derived
gradient, fit, exact score pmf, walk-forward). Tests: `tests/test_bivariate.py` (gradient against finite
differences; reduces to Model B when lam3 -> 0; pmf sums to 1 with the right means and covariance; closed form
equals the convolution grid; recovery of lam3 = 0.3 from 20,000 simulated matches).

Setup: same as Model B, xi = 1.0, kappa = 5, weekly refits using only earlier matches, same matches in both models.

| Window | Matches | Log loss, Model B | Log loss, bivariate | Difference | Score log-lik, B | bivariate | Mean fitted lam3 |
|---|---|---|---|---|---|---|---|
| Early 2018-21 | 1,140 | 0.96247 | 0.96247 | 0.00000 | -2.91366 | -2.91367 | 0.0001 |
| Validation 2021-23 | 760 | 0.97361 | 0.97361 | 0.00000 | -2.96597 | -2.96597 | 0.0001 |
| Test 2023-26 | 1,140 | 0.98523 | 0.98523 | 0.00000 | -2.98248 | -2.98248 | 0.0001 |

Model B reproduces the earlier reported numbers (validation 0.9736, test 0.9852), which checks the harness.

Result: the fitted shared rate sits on its lower bound (0.0001) in every weekly refit, so the bivariate model is
the independent model. This is a genuine optimum, not a failed fit: with the other parameters re-optimised, the
penalised log-likelihood on the 4,260 matches before 31 Oct is -1004.54 at lam3 = 0.0001, -1005.16 at 0.05,
-1006.18 at 0.1 and -1009.35 at 0.2 (falling steadily). The time-weighted correlation of the residuals
(goals minus fitted rate) is 0.005.

Why: once team strengths and home advantage are fitted, home and away goals show no positive dependence. The
earlier Dixon-Coles fit (rho = -0.09) points the other way, a small negative dependence at low scores, which a
bivariate Poisson cannot express (it allows Cov >= 0 only). Not adopted.

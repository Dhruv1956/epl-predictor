# Walk-forward results: independent Poisson model

Validation window: 2021-07-01 to 2023-07-01. Test window: 2023-07-01 to 2026-07-01.
Chosen on validation: xi = 1.0 per year (half-life 0.7 years), kappa = 5.0.

## Test window: each predictor on the matches it has data for
| predictor | matches | log_loss | brier | accuracy | model_on_same_matches |
|---|---|---|---|---|---|
| Poisson model | 1140 | 0.9852 | 0.5867 | 52.0% | nan |
| Uniform 1/3 | 1140 | 1.0986 | 0.6667 | 43.2% | nan |
| Training class frequencies | 1140 | 1.0728 | 0.6495 | 43.2% | nan |
| B365 (margin removed) | 1140 | 0.9662 | 0.5744 | 54.2% | 0.9852 |
| Avg (margin removed) | 1140 | 0.9650 | 0.5740 | 54.2% | 0.9852 |
| PS (margin removed) | 970 | 0.9501 | 0.5635 | 55.6% | 0.9710 |

## Test window: matches where every odds source is present
| predictor | matches | log_loss | brier | accuracy |
|---|---|---|---|---|
| Poisson model | 970 | 0.9710 | 0.5766 | 54.0% |
| Uniform 1/3 | 970 | 1.0986 | 0.6667 | 43.8% |
| Training class frequencies | 970 | 1.0682 | 0.6465 | 43.8% |
| B365 (margin removed) | 970 | 0.9510 | 0.5638 | 55.8% |
| Avg (margin removed) | 970 | 0.9503 | 0.5636 | 55.7% |
| PS (margin removed) | 970 | 0.9501 | 0.5635 | 55.6% |

## Validation grid (log loss)
| xi | kappa | log_loss | brier | accuracy |
|---|---|---|---|---|
| 0.0 | 1.0 | 0.9883 | 0.5883 | 51.2% |
| 0.0 | 5.0 | 0.9858 | 0.5866 | 51.3% |
| 0.0 | 20.0 | 0.9842 | 0.5853 | 52.0% |
| 0.2 | 1.0 | 0.9815 | 0.5838 | 52.1% |
| 0.2 | 5.0 | 0.9792 | 0.5823 | 52.4% |
| 0.2 | 20.0 | 0.9798 | 0.5824 | 52.6% |
| 0.5 | 1.0 | 0.9773 | 0.5809 | 52.6% |
| 0.5 | 5.0 | 0.9752 | 0.5795 | 52.5% |
| 0.5 | 20.0 | 0.9801 | 0.5824 | 52.8% |
| 1.0 | 1.0 | 0.9753 | 0.5796 | 54.3% |
| 1.0 | 5.0 | 0.9736 | 0.5783 | 54.1% |
| 1.0 | 20.0 | 0.9861 | 0.5864 | 53.4% |

## Fitted strengths on all data (teams active in the last 400 days)
iota = 0.243, eta = 0.182

| team | attack | defence |
|---|---|---|
| Man City | 0.391 | 0.337 |
| Arsenal | 0.252 | 0.447 |
| Liverpool | 0.211 | 0.107 |
| Chelsea | 0.202 | -0.009 |
| Brighton | 0.197 | 0.131 |
| Man United | 0.145 | 0.063 |
| Newcastle | 0.142 | -0.049 |
| Brentford | 0.125 | 0.074 |
| Aston Villa | 0.076 | 0.03 |
| Bournemouth | 0.055 | 0.005 |
| Hull | -0.008 | 0.145 |
| Leeds | -0.019 | 0.096 |
| Nott'm Forest | -0.045 | 0.078 |
| West Ham | -0.055 | -0.086 |
| Crystal Palace | -0.063 | -0.012 |
| Tottenham | -0.064 | -0.057 |
| Ipswich | -0.073 | -0.266 |
| Fulham | -0.088 | 0.038 |
| Everton | -0.095 | 0.142 |
| Coventry | -0.105 | -0.044 |
| Sunderland | -0.14 | 0.022 |
| Burnley | -0.202 | -0.219 |
| Wolves | -0.269 | -0.121 |

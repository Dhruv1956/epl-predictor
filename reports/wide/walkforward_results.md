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
| 0.5 | 2.0 | 0.9762 | 0.5802 | 52.5% |
| 0.5 | 5.0 | 0.9752 | 0.5795 | 52.5% |
| 0.5 | 10.0 | 0.9759 | 0.5798 | 52.9% |
| 1.0 | 2.0 | 0.9740 | 0.5787 | 54.3% |
| 1.0 | 5.0 | 0.9736 | 0.5783 | 54.1% |
| 1.0 | 10.0 | 0.9769 | 0.5803 | 53.3% |
| 1.5 | 2.0 | 0.9738 | 0.5788 | 53.8% |
| 1.5 | 5.0 | 0.9742 | 0.5788 | 55.0% |
| 1.5 | 10.0 | 0.9799 | 0.5823 | 54.3% |
| 2.0 | 2.0 | 0.9747 | 0.5795 | 54.5% |
| 2.0 | 5.0 | 0.9761 | 0.5801 | 54.6% |
| 2.0 | 10.0 | 0.9838 | 0.5849 | 54.5% |
| 3.0 | 2.0 | 0.9786 | 0.5823 | 53.8% |
| 3.0 | 5.0 | 0.9816 | 0.5839 | 54.3% |
| 3.0 | 10.0 | 0.9925 | 0.5910 | 53.8% |

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

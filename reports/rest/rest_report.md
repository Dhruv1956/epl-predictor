# Rest-days feature (Model B base, xi=1, kappa=5)

Rest = days since a team's previous Premier League match in the data (earlier dates only). Cup and European games are
not in the data, so this is a lower bound on tiredness. One shared coefficient gamma: log lambda += gamma z, log mu -= gamma z.

Features: linear z = (min(rest_h,7) - min(rest_a,7))/3 ; short z = 1[rest_h<=3] - 1[rest_a<=3]. 12.5% of team-matches have rest <= 3 days.

Paired change in log loss against Model B (negative = better), weekly walk-forward:

| feature | window | n | delta | SE | gamma (mean of weekly fits) |
|---|---|---|---|---|---|
| linear | early | 1140 | +0.0014 | 0.0008 | -0.047 |
| linear | validation | 760 | +0.0015 | 0.0013 | +0.038 |
| linear | test | 1140 | -0.0015 | 0.0011 | -0.085 |
| short | early | 1140 | -0.0026 | 0.0024 | +0.153 |
| short | validation | 760 | -0.0012 | 0.0013 | +0.072 |
| short | test | 1140 | -0.0011 | 0.0017 | +0.114 |

Reading: no change exceeds 1.2 SE and the linear coefficient changes sign between windows. The short-rest flag has
a positive gamma in the early and validation windows and on average in the test window: a side on short rest scored
slightly MORE, not less. That is the opposite of fatigue. A likely reason is selection: sides with short gaps are
mostly those playing in Europe, which are the stronger sides, so the flag partly stands in for strength that the
shrunken ratings under-state. The gain is 0.001 to 0.003 per match, 0.9 SE on validation, so it would not be selected
and is not adopted. The SEs understate team-level uncertainty.

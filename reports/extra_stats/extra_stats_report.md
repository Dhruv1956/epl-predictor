# Total shots and corners on top of v2-shots

Each extra statistic (HS/AS total shots, HC/AC corners) gets its own stage-0 Poisson fit on earlier matches only (xi_s=1.0, kappa_s=5) and its own coefficient, added to v2-shots (xi=1.5, kappa=5).
Paired change in log loss against v2-shots (positive = worse), weekly walk-forward:

| variant | early | validation | test |
|---|---|---|---|
| + total shots | +0.00052 (SE 0.00045) | +0.00101 (0.00050) | +0.00034 (0.00015) |
| + corners | +0.00018 (0.00027) | +0.00015 (0.00025) | +0.00015 (0.00021) |
| + shots and corners | +0.00063 (0.00053) | +0.00130 (0.00063) | +0.00021 (0.00029) |

Mean fitted coefficients: shots on target 1.4 to 1.8 in every variant. Total shots: -0.35, -0.43, +0.06 (early, validation, test); corners: -0.04, -0.16, +0.02. Total shots flip sign in the test window, so there is no stable effect.
Reading: given shots on target, total shots and corners add nothing; every variant is slightly worse. Not adopted. Free columns in the football-data files are exhausted as a source of extra information (fouls, cards and half-time scores are not team-quality measures); the remaining gap to the bookmakers is information they have that the file does not (line-ups, injuries, xG).

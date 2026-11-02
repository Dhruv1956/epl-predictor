# Stacking the shots model (v2) with an Elo + ordered-logit model

Elo: start 1500, home advantage 65 points in the update, update size K, ratings pulled back by `carry` after any gap over 30 days.
Link: ordered logit on d = R_home - R_away, refitted weekly on earlier matches (hand gradient, checked numerically).
Elo settings chosen on validation by the Elo model's own log loss: K=30, carry=1.0 (K=10 to 40 and carry 0.75 / 1.0 tried).

Log loss by window:

| model | early | validation | test |
|---|---|---|---|
| shots Poisson (v2) | 0.9601 | 0.9740 | 0.9770 |
| Elo alone | 0.9745 | 0.9658 | 0.9845 |

Linear pool p = (1-w) p_shots + w p_elo, log loss by window:

| w | early | validation | test | early+validation |
|---|---|---|---|---|
| 0.0 | 0.96010 | 0.97402 | 0.97696 | 0.96567 |
| 0.1 | 0.95981 | 0.97199 | 0.97651 | 0.96469 |
| 0.2 | 0.95990 | 0.97027 | 0.97632 | 0.96405 |
| 0.3 | 0.96036 | 0.96883 | 0.97640 | 0.96375 |
| 0.5 | 0.96239 | 0.96674 | 0.97732 | 0.96413 |
| 0.8 | 0.96834 | 0.96545 | 0.98075 | 0.96719 |
| 1.0 | 0.97447 | 0.96582 | 0.98451 | 0.97101 |

Chosen on validation alone (project rule): w = 0.8 -> delta vs v2: early +0.0082 (SE 0.0046), test +0.0038 (SE 0.0039).
Log pool (a=0.16 on shots, b=0.82 on Elo, fitted on validation): early +0.0087, test +0.0043 (SE 0.004-0.005).
Chosen on early+validation (post-hoc, no test data used): w = 0.3 -> test -0.0006 (SE 0.0014).

Reading: Elo wins only in the validation window (2021-23), the stretch where the Poisson models also trailed the bookmakers most, so a weight fitted
there is too large. A small weight helps a little everywhere but is far inside the noise. Not adopted. Worth knowing: a fast-adapting
results-only model does better than a slowly-learning Poisson model in 2021-23, which is a clue about what changed in those seasons.

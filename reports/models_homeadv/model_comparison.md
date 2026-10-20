# Model comparison (test window)

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_ref | se_ref | dLL_vs_Avg | se_Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| working: cd + promoted (2-stage) | 1.0000 | 5.0000 | 1140 | 0.9815 | 0.5845 | 0.5246 | 0.0000 | 0.0000 | 0.0165 | 0.0046 |
| working + team home adv (kappa_h=20) | 1.0000 | 5.0000 | 1140 | 0.9805 | 0.5838 | 0.5263 | -0.0010 | 0.0008 | 0.0155 | 0.0046 |
| working + team home adv (kappa_h=60) | 1.0000 | 5.0000 | 1140 | 0.9810 | 0.5842 | 0.5228 | -0.0005 | 0.0003 | 0.0160 | 0.0046 |
| working + team home adv (kappa_h=5) | 0.7500 | 5.0000 | 1140 | 0.9794 | 0.5832 | 0.5246 | -0.0021 | 0.0018 | 0.0144 | 0.0048 |
| Avg odds (margin removed) | nan | nan | 1140 | 0.9650 | 0.5740 | 0.5421 | nan | nan | nan | nan |

## Split by subset

| window | subset | matches | model | ref_log_loss | log_loss | dLL_vs_ref | se |
|---|---|---|---|---|---|---|---|
| validation | all matches | 760 | working + team home adv (kappa_h=20) | 0.9733 | 0.9736 | 0.0003 | 0.0009 |
| validation | all matches | 760 | working + team home adv (kappa_h=60) | 0.9733 | 0.9733 | 0.0000 | 0.0004 |
| validation | all matches | 760 | working + team home adv (kappa_h=5) | 0.9733 | 0.9746 | 0.0013 | 0.0021 |
| validation | involving a newly promoted team | 216 | working + team home adv (kappa_h=20) | 0.9184 | 0.9174 | -0.0010 | 0.0017 |
| validation | involving a newly promoted team | 216 | working + team home adv (kappa_h=60) | 0.9184 | 0.9179 | -0.0005 | 0.0007 |
| validation | involving a newly promoted team | 216 | working + team home adv (kappa_h=5) | 0.9184 | 0.9164 | -0.0020 | 0.0042 |
| validation | all other matches | 544 | working + team home adv (kappa_h=20) | 0.9951 | 0.9959 | 0.0008 | 0.0010 |
| validation | all other matches | 544 | working + team home adv (kappa_h=60) | 0.9951 | 0.9953 | 0.0003 | 0.0004 |
| validation | all other matches | 544 | working + team home adv (kappa_h=5) | 0.9951 | 0.9977 | 0.0026 | 0.0024 |
| test | all matches | 1140 | working + team home adv (kappa_h=20) | 0.9815 | 0.9805 | -0.0010 | 0.0008 |
| test | all matches | 1140 | working + team home adv (kappa_h=60) | 0.9815 | 0.9810 | -0.0005 | 0.0003 |
| test | all matches | 1140 | working + team home adv (kappa_h=5) | 0.9815 | 0.9794 | -0.0021 | 0.0018 |
| test | involving a newly promoted team | 324 | working + team home adv (kappa_h=20) | 0.8865 | 0.8844 | -0.0020 | 0.0014 |
| test | involving a newly promoted team | 324 | working + team home adv (kappa_h=60) | 0.8865 | 0.8856 | -0.0009 | 0.0006 |
| test | involving a newly promoted team | 324 | working + team home adv (kappa_h=5) | 0.8865 | 0.8803 | -0.0062 | 0.0035 |
| test | all other matches | 816 | working + team home adv (kappa_h=20) | 1.0192 | 1.0186 | -0.0006 | 0.0009 |
| test | all other matches | 816 | working + team home adv (kappa_h=60) | 1.0192 | 1.0189 | -0.0003 | 0.0004 |
| test | all other matches | 816 | working + team home adv (kappa_h=5) | 1.0192 | 1.0187 | -0.0005 | 0.0022 |

## Fitted extras

- working: cd + promoted (2-stage): eta = 0.183; closed_doors = -0.152; promoted_attack = -0.172; promoted_defence = -0.107
- working + team home adv (kappa_h=20): eta = 0.175; closed_doors = -0.155; promoted_attack = -0.172; promoted_defence = -0.107; delta sd = 0.051; lowest Nott'm Forest -0.14, Bournemouth -0.05, Sunderland -0.05; highest Fulham +0.05, Man City +0.06, Newcastle +0.10
- working + team home adv (kappa_h=60): eta = 0.180; closed_doors = -0.153; promoted_attack = -0.172; promoted_defence = -0.107; delta sd = 0.020; lowest Nott'm Forest -0.06, Bournemouth -0.02, Sunderland -0.02; highest Fulham +0.02, Man City +0.03, Newcastle +0.04
- working + team home adv (kappa_h=5): eta = 0.154; closed_doors = -0.157; promoted_attack = -0.172; promoted_defence = -0.107; delta sd = 0.119; lowest Nott'm Forest -0.31, Ipswich -0.14, Bournemouth -0.14; highest Fulham +0.13, Man City +0.14, Newcastle +0.24

# Model comparison (test window)

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_ref | se_ref | dLL_vs_Avg | se_Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| working: cd + promoted (2-stage) | 1.0000 | 5.0000 | 1140 | 0.9815 | 0.5845 | 0.5246 | 0.0000 | 0.0000 | 0.0165 | 0.0046 |
| working + shots (xi_s=0.3) | 2.0000 | 5.0000 | 1140 | 0.9804 | 0.5843 | 0.5237 | -0.0011 | 0.0026 | 0.0154 | 0.0044 |
| working + shots (xi_s=0.6) | 1.5000 | 5.0000 | 1140 | 0.9772 | 0.5820 | 0.5202 | -0.0043 | 0.0024 | 0.0122 | 0.0045 |
| working + shots (xi_s=1.0) | 1.5000 | 5.0000 | 1140 | 0.9770 | 0.5817 | 0.5254 | -0.0045 | 0.0025 | 0.0119 | 0.0046 |
| Avg odds (margin removed) | nan | nan | 1140 | 0.9650 | 0.5740 | 0.5421 | nan | nan | nan | nan |

## Split by subset

| window | subset | matches | model | ref_log_loss | log_loss | dLL_vs_ref | se |
|---|---|---|---|---|---|---|---|
| validation | all matches | 760 | working + shots (xi_s=0.3) | 0.9733 | 0.9746 | 0.0013 | 0.0041 |
| validation | all matches | 760 | working + shots (xi_s=0.6) | 0.9733 | 0.9747 | 0.0014 | 0.0038 |
| validation | all matches | 760 | working + shots (xi_s=1.0) | 0.9733 | 0.9740 | 0.0007 | 0.0039 |
| validation | involving a newly promoted team | 216 | working + shots (xi_s=0.3) | 0.9184 | 0.9232 | 0.0048 | 0.0093 |
| validation | involving a newly promoted team | 216 | working + shots (xi_s=0.6) | 0.9184 | 0.9244 | 0.0060 | 0.0094 |
| validation | involving a newly promoted team | 216 | working + shots (xi_s=1.0) | 0.9184 | 0.9247 | 0.0063 | 0.0094 |
| validation | all other matches | 544 | working + shots (xi_s=0.3) | 0.9951 | 0.9950 | -0.0001 | 0.0043 |
| validation | all other matches | 544 | working + shots (xi_s=0.6) | 0.9951 | 0.9946 | -0.0005 | 0.0038 |
| validation | all other matches | 544 | working + shots (xi_s=1.0) | 0.9951 | 0.9936 | -0.0015 | 0.0040 |
| test | all matches | 1140 | working + shots (xi_s=0.3) | 0.9815 | 0.9804 | -0.0011 | 0.0026 |
| test | all matches | 1140 | working + shots (xi_s=0.6) | 0.9815 | 0.9772 | -0.0043 | 0.0024 |
| test | all matches | 1140 | working + shots (xi_s=1.0) | 0.9815 | 0.9770 | -0.0045 | 0.0025 |
| test | involving a newly promoted team | 324 | working + shots (xi_s=0.3) | 0.8865 | 0.8810 | -0.0055 | 0.0063 |
| test | involving a newly promoted team | 324 | working + shots (xi_s=0.6) | 0.8865 | 0.8742 | -0.0123 | 0.0057 |
| test | involving a newly promoted team | 324 | working + shots (xi_s=1.0) | 0.8865 | 0.8735 | -0.0130 | 0.0058 |
| test | all other matches | 816 | working + shots (xi_s=0.3) | 1.0192 | 1.0199 | 0.0007 | 0.0027 |
| test | all other matches | 816 | working + shots (xi_s=0.6) | 1.0192 | 1.0182 | -0.0011 | 0.0024 |
| test | all other matches | 816 | working + shots (xi_s=1.0) | 1.0192 | 1.0180 | -0.0012 | 0.0025 |

## Fitted extras

- working: cd + promoted (2-stage): eta = 0.187; closed_doors = -0.152; promoted_attack = -0.170; promoted_defence = -0.108
- working + shots (xi_s=0.3): eta = 0.207; closed_doors = -0.162; promoted_attack = -0.161; promoted_defence = -0.103; shots_strength = 1.353
- working + shots (xi_s=0.6): eta = 0.198; closed_doors = -0.170; promoted_attack = -0.167; promoted_defence = -0.105; shots_strength = 1.552
- working + shots (xi_s=1.0): eta = 0.198; closed_doors = -0.166; promoted_attack = -0.169; promoted_defence = -0.107; shots_strength = 1.608

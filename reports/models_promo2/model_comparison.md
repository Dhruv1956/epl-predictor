# Model comparison (test window)

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_ref | se_ref | dLL_vs_Avg | se_Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| B + closed doors | 1.0000 | 5.0000 | 1140 | 0.9851 | 0.5867 | 0.5228 | 0.0000 | 0.0000 | 0.0201 | 0.0047 |
| promoted, joint | 0.7500 | 5.0000 | 1140 | 0.9789 | 0.5833 | 0.5298 | -0.0062 | 0.0026 | 0.0139 | 0.0048 |
| promoted, offsets xi=0 | 1.0000 | 5.0000 | 1140 | 0.9815 | 0.5845 | 0.5246 | -0.0036 | 0.0012 | 0.0165 | 0.0046 |
| promoted, offsets xi=0.3 | 1.0000 | 5.0000 | 1140 | 0.9805 | 0.5840 | 0.5219 | -0.0046 | 0.0017 | 0.0155 | 0.0046 |
| Avg odds (margin removed) | nan | nan | 1140 | 0.9650 | 0.5740 | 0.5421 | nan | nan | nan | nan |

## Split by subset

| window | subset | matches | model | ref_log_loss | log_loss | dLL_vs_ref | se |
|---|---|---|---|---|---|---|---|
| validation | all matches | 760 | promoted, joint | 0.9733 | 0.9746 | 0.0013 | 0.0022 |
| validation | all matches | 760 | promoted, offsets xi=0 | 0.9733 | 0.9733 | 0.0000 | 0.0006 |
| validation | all matches | 760 | promoted, offsets xi=0.3 | 0.9733 | 0.9735 | 0.0002 | 0.0010 |
| validation | involving a newly promoted team | 216 | promoted, joint | 0.9191 | 0.9184 | -0.0007 | 0.0064 |
| validation | involving a newly promoted team | 216 | promoted, offsets xi=0 | 0.9191 | 0.9184 | -0.0007 | 0.0018 |
| validation | involving a newly promoted team | 216 | promoted, offsets xi=0.3 | 0.9191 | 0.9182 | -0.0009 | 0.0030 |
| validation | all other matches | 544 | promoted, joint | 0.9948 | 0.9969 | 0.0022 | 0.0017 |
| validation | all other matches | 544 | promoted, offsets xi=0 | 0.9948 | 0.9951 | 0.0003 | 0.0005 |
| validation | all other matches | 544 | promoted, offsets xi=0.3 | 0.9948 | 0.9955 | 0.0007 | 0.0008 |
| test | all matches | 1140 | promoted, joint | 0.9851 | 0.9789 | -0.0062 | 0.0026 |
| test | all matches | 1140 | promoted, offsets xi=0 | 0.9851 | 0.9815 | -0.0036 | 0.0012 |
| test | all matches | 1140 | promoted, offsets xi=0.3 | 0.9851 | 0.9805 | -0.0046 | 0.0017 |
| test | involving a newly promoted team | 324 | promoted, joint | 0.9004 | 0.8752 | -0.0252 | 0.0080 |
| test | involving a newly promoted team | 324 | promoted, offsets xi=0 | 0.9004 | 0.8865 | -0.0140 | 0.0038 |
| test | involving a newly promoted team | 324 | promoted, offsets xi=0.3 | 0.9004 | 0.8821 | -0.0183 | 0.0054 |
| test | all other matches | 816 | promoted, joint | 1.0188 | 1.0201 | 0.0013 | 0.0016 |
| test | all other matches | 816 | promoted, offsets xi=0 | 1.0188 | 1.0192 | 0.0005 | 0.0007 |
| test | all other matches | 816 | promoted, offsets xi=0.3 | 1.0188 | 1.0196 | 0.0009 | 0.0010 |

## Fitted extras

- B + closed doors: eta = 0.183; closed_doors = -0.156
- promoted, joint: eta = 0.180; closed_doors = -0.151; promoted_attack = -0.246; promoted_defence = -0.200
- promoted, offsets xi=0: eta = 0.183; closed_doors = -0.152; promoted_attack = -0.172; promoted_defence = -0.107
- promoted, offsets xi=0.3: eta = 0.183; closed_doors = -0.151; promoted_attack = -0.215; promoted_defence = -0.153

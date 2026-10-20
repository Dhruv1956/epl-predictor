# Model comparison (test window)

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_ref | se_ref | dLL_vs_Avg | se_Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| B + closed doors | 1.0000 | 5.0000 | 1140 | 0.9851 | 0.5867 | 0.5228 | 0.0000 | 0.0000 | 0.0201 | 0.0047 |
| B + cd + promoted | 0.7500 | 5.0000 | 1140 | 0.9789 | 0.5833 | 0.5298 | -0.0062 | 0.0026 | 0.0139 | 0.0048 |
| C + promoted | 0.7500 | 5.0000 | 1140 | 0.9782 | 0.5829 | 0.5298 | -0.0069 | 0.0026 | 0.0132 | 0.0048 |
| Avg odds (margin removed) | nan | nan | 1140 | 0.9650 | 0.5740 | 0.5421 | nan | nan | nan | nan |

## Split by subset

| subset | matches | model | ref_log_loss | log_loss | dLL_vs_ref | se |
|---|---|---|---|---|---|---|
| involving a newly promoted team | 324 | B + cd + promoted | 0.9004 | 0.8752 | -0.0252 | 0.0080 |
| involving a newly promoted team | 324 | C + promoted | 0.9004 | 0.8746 | -0.0258 | 0.0080 |
| all other matches | 816 | B + cd + promoted | 1.0188 | 1.0201 | 0.0013 | 0.0016 |
| all other matches | 816 | C + promoted | 1.0188 | 1.0194 | 0.0006 | 0.0017 |

## Fitted extras

- B + closed doors: eta = 0.183; closed_doors = -0.156
- B + cd + promoted: eta = 0.180; closed_doors = -0.151; promoted_attack = -0.246; promoted_defence = -0.200
- C + promoted: eta = 0.178; closed_doors = -0.150; promoted_attack = -0.247; promoted_defence = -0.199; rho = -0.087

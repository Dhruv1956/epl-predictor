# Model comparison (test window)

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_B | se_B | dLL_vs_Avg | se_Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| B: Poisson | 1.0000 | 5.0000 | 1140 | 0.9852 | 0.5867 | 0.5202 | 0.0000 | 0.0000 | 0.0202 | 0.0047 |
| B + closed doors | 1.0000 | 5.0000 | 1140 | 0.9851 | 0.5867 | 0.5228 | -0.0001 | 0.0002 | 0.0201 | 0.0047 |
| B + Dixon-Coles | 1.0000 | 5.0000 | 1140 | 0.9845 | 0.5863 | 0.5193 | -0.0007 | 0.0008 | 0.0195 | 0.0047 |
| C: both | 1.0000 | 5.0000 | 1140 | 0.9844 | 0.5863 | 0.5219 | -0.0008 | 0.0008 | 0.0194 | 0.0047 |
| Avg odds (margin removed) | nan | nan | 1140 | 0.9650 | 0.5740 | 0.5421 | nan | nan | nan | nan |

## Fitted extras

- B + closed doors: eta (with fans) = 0.183; gamma = -0.156; eta + gamma (no fans) = 0.027; rho = 0.000
- C: both: eta (with fans) = 0.180; gamma = -0.155; eta + gamma (no fans) = 0.025; rho = -0.096
- B + Dixon-Coles: rho = -0.096

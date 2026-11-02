# Drifting strengths under the v2-shots covariates

Same covariates as v2-shots (closed doors, promoted offsets, stage-0 shot strengths with xi_s=1.0), but team strengths
drift by 91-day block. Promoted offsets are estimated jointly (v2 uses a separate first stage). kappa0=5. Grid on validation:

| xi | k1 | validation LL | delta vs v2 | SE |
|---|---|---|---|---|
| 0 | 30 | 0.97487 | +0.0008 | 0.0032 |
| 0 | 100 | 0.97306 | -0.0010 | 0.0026 |
| 0 | 300 | 0.97565 | +0.0016 | 0.0044 |
| 0.5 | 30 | 0.97686 | +0.0028 | 0.0030 |
| 0.5 | 100 | 0.97450 | +0.0005 | 0.0019 |
| 0.5 | 300 | 0.97509 | +0.0011 | 0.0028 |

Best (xi=0, k1=100) against v2-shots (xi=1.5, kappa=5):

| window | n | LL v2 | LL drift+shots | delta | SE |
|---|---|---|---|---|---|
| early | 1140 | 0.9601 | 0.9596 | -0.0005 | 0.0025 |
| validation | 760 | 0.9740 | 0.9731 | -0.0010 | 0.0026 |
| test | 1140 | 0.9770 | 0.9800 | +0.0030 | 0.0022 |

Reading: the gain seen without shots (-0.0014, -0.0019, -0.0044) disappears and the test window turns worse. The
shots covariate already carries the information that drifting strengths were adding. Not adopted; no v3.

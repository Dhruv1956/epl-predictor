# Calibration and ridge-strength experiments (2 Nov 2026)

Code: `src/improve.py`, tests `tests/test_improve.py` (recalibration gradient against finite differences, recovery of a known
distortion, split ridge equals Model B when the two strengths are equal). Weekly walk-forward, only earlier matches,
windows early 2018-21, validation 2021-23 (all choices), test 2023-26. Differences are paired log loss, new minus old
(negative = better), with standard errors that treat matches as independent.

## 1. Calibration (Model B, xi = 1.0, kappa = 5)
Recalibration z_k = t log p_k + b_k, fitted on validation: t = 1.115, b_H = 0.138, b_D = 0.110. Applied unchanged:
early +0.0031 +- 0.0023 (worse), validation -0.0029 +- 0.0027 (in sample), test +0.0001 +- 0.0022. The window-by-window
fits of the home and draw shifts change sign (b_H -0.15, +0.14, -0.04), so there is no stable bias. Not adopted.
Temperature only (b = 0), t = 1.12 from validation: -0.0018 +- 0.0015, -0.0013 +- 0.0018, -0.0009 +- 0.0014 (pooled about
-0.0013 +- 0.0009): the plain Poisson model is slightly too flat (under-confident).
Same test on the working model v1: t = 1.085, pooled -0.0010 +- 0.0006. On the headline model v2 (shots, xi = 1.5):
t = 0.92 on validation, window fits 0.995, 0.919, 0.979, pooled +0.0002 +- 0.0007: v2 is already well calibrated.

## 2. Separate ridge strength for attack and defence (Model B, xi = 1.0)
Validation grid over kappa_a, kappa_b in {2, 3.5, 5, 8, 12, 20}: best (3.5, 3.5); the surface is flat along the
kappa_a / kappa_b direction (for example (5, 2) 0.97340 and (5, 3.5) 0.97339 against 0.97338 at (3.5, 3.5)).
Attack and defence prefer the same strength, so separating them gains nothing.

## 3. A side finding: one kappa, retuned
Equal kappa, paired against kappa = 5: kappa 3.5 gives early -0.0015 +- 0.0007, validation -0.0002 +- 0.0008,
test -0.0018 +- 0.0006 (pooled -0.0013 +- 0.0004). The validation curve is flat between 3 and 4, so validation alone
does not distinguish them; the other windows favour lower kappa. The same holds for v1 (pooled -0.0010 +- 0.0004).
It does NOT hold for the headline model v2 (shots, xi = 1.5): kappa 3.5 is worse than 5 in all windows
(+0.0004, +0.0007, +0.0006; pooled +0.0006 +- 0.0002). The shots stage already supplies the prior mean for team strength,
so v2 needs more shrinkage, not less. Plain Poisson at xi = 1.0 is slightly over-shrunk; v2 at kappa = 5 is not.

## Decision
No change to any registered model version. The live versions stay frozen. Neither recalibration nor separate ridge
strengths helps the headline model, and the kappa gain for the older versions (about 0.001 of a 0.012 gap) does not
carry over to it.

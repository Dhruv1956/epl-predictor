# Drifting team strengths vs Model B (xi=1, kappa=5)

Per-block (91-day) attack and defence strengths with a random-walk prior (k0=5 on the first block, k1 on block-to-block
changes); forecasts use the last block. Weekly walk-forward. Stiffness k1 and decay xi chosen on validation only.

Validation grid (xi=0): k1=30 +0.0006, k1=100 -0.0019, k1=300 +0.0004, k1=1000 +0.0051, k1=3000 +0.0089 (change in log loss vs B).
With xi=0.5 every value was worse than xi=0. Low k1 (3, 10) is clearly worse: too much drift.

| window | n | LL B | LL drift | delta | SE |
|---|---|---|---|---|---|
| early | 1140 | 0.9625 | 0.9611 | -0.0014 | 0.0028 |
| validation | 760 | 0.9736 | 0.9718 | -0.0019 | 0.0033 |
| test | 1140 | 0.9852 | 0.9808 | -0.0044 | 0.0024 |

By season (date-based seasons): 2018/19 -0.0089, 2019/20 -0.0073, 2020/21 +0.0093, 2021/22 -0.0062, 2022/23 +0.0025,
2023/24 -0.0110, 2024/25 -0.0028, 2025/26 +0.0006 (SE 0.003 to 0.006 each).

Reading: same sign in all windows, 6 of 8 seasons better, but small and not beyond 1.8 SE; chosen from 12 configurations;
the shots model (test 0.9770) is still ahead. Gap to bookmakers on test: 0.0158 vs 0.0202 for B (book 0.9650).
Next: put drifting strengths under the shots model.

# Premier League + Championship joint model

Cells = (team, 91-day period inside a season). League-relative strengths A, B per cell; random-walk prior on the ABSOLUTE strength
(A + m_a f, B + m_d f; f=1 in the Championship) links a team's consecutive cells, so promoted/relegated clubs (27 seen in both leagues) identify the
league gap m. Championship goal level and home advantage get their own offsets (g1, g2). k0=5; stiffness k1 chosen on validation.
Data: 6,228 Championship matches (2015/16 to 2026/27) from football-data.co.uk E1 files.

Validation grid (change in log loss vs Model B): k1=50 -0.0037, k1=100 -0.0039, k1=200 -0.0028. Chosen k1=100.

| window | n | vs | delta | SE | promoted-club matches | delta (promoted) | SE |
|---|---|---|---|---|---|---|---|
| early | 1140 | Model B | -0.0010 | 0.0028 | 324 | +0.0035 | 0.0069 |
| early | 1140 | single-league drift | +0.0004 | 0.0019 | 324 | +0.0070 | 0.0064 |
| validation | 760 | Model B | -0.0039 | 0.0034 | 216 | -0.0078 | 0.0082 |
| validation | 760 | single-league drift | -0.0021 | 0.0028 | 216 | -0.0056 | 0.0097 |
| test | 1140 | Model B | -0.0004 | 0.0036 | 324 | -0.0030 | 0.0113 |
| test | 1140 | single-league drift | +0.0040 | 0.0026 | 324 | +0.0138 | 0.0092 |

Learned league gap (last refit in each window): m_a = -0.51 / -0.44 / -0.40, m_d = -0.31 / -0.25 / -0.30 (Championship attack and defence weaker than the Premier League average).

Reading: the gap is stable and sensible, but the forecasts do not improve: nothing beyond 1.2 SE against Model B, and worse than the Premier-League-only drift model on test (+0.0040; promoted-club matches +0.0138, 1.5 SE). The second-tier ratings of promoted clubs carry no extra predictive information
beyond what the first few Premier League matches already give. Not adopted.

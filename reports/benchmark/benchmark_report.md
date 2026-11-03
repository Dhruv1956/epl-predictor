# Closing-odds benchmark and ranked probability score

Matches with every price available (Avg, Pinnacle, Bet365, before and closing); closing odds exist from 2019/20 and Pinnacle odds
stop partway through 2025/26, so the test window has 970 of 1140 matches. All margins removed by normalisation.

| window | n | source | log loss | Brier | RPS | LL minus v2 | SE |
|---|---|---|---|---|---|---|---|
| early | 760 | Model B | 0.9940 | 0.5917 | 0.2081 | -0.0018 | 0.0040 |
| early | 760 | v2-shots | 0.9959 | 0.5916 | 0.2080 | 0 | |
| early | 760 | Avg odds | 0.9884 | 0.5859 | 0.2059 | -0.0075 | 0.0062 |
| early | 760 | Avg closing | 0.9846 | 0.5834 | 0.2047 | -0.0113 | 0.0067 |
| validation | 760 | Model B | 0.9736 | 0.5783 | 0.2014 | -0.0004 | 0.0042 |
| validation | 760 | v2-shots | 0.9740 | 0.5787 | 0.2014 | 0 | |
| validation | 760 | Avg odds | 0.9509 | 0.5637 | 0.1940 | -0.0231 | 0.0058 |
| validation | 760 | Avg closing | 0.9494 | 0.5627 | 0.1932 | -0.0247 | 0.0063 |
| test | 970 | Model B | 0.9710 | 0.5766 | 0.1987 | +0.0090 | 0.0033 |
| test | 970 | v2-shots | 0.9619 | 0.5711 | 0.1961 | 0 | |
| test | 970 | Avg odds | 0.9503 | 0.5636 | 0.1924 | -0.0117 | 0.0051 |
| test | 970 | Avg closing | 0.9445 | 0.5591 | 0.1906 | -0.0174 | 0.0054 |

(Pinnacle and Bet365, before and closing, are in benchmark.csv; they are within 0.001 of the Avg rows.)

Reading: closing odds are about 0.006 harder to beat than the earlier-recorded odds (test gap 0.0174 against 0.0117). The ordering of all sources is
the same under log loss, Brier and RPS, so the conclusions do not depend on the scoring rule. On the early window the shots model is actually slightly behind Model B on this subset
(2019/20 onward); on the test subset it leads Model B by 0.009 (2.7 SE), consistent with the main results. The test numbers differ from the headline (0.9770) only because this is a subset.

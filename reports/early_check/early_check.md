# Early-window check

| model | xi | kappa | matches | log_loss | brier | accuracy | dLL_vs_working | se | dLL_vs_B365 | se_B365 |
|---|---|---|---|---|---|---|---|---|---|---|
| B: Poisson | 1.0000 | 5.0000 | 1140 | 0.9625 | 0.5698 | 0.5447 | -0.0011 | 0.0011 | 0.0048 | 0.0050 |
| B + closed doors | 1.0000 | 5.0000 | 1140 | 0.9628 | 0.5701 | 0.5474 | -0.0008 | 0.0007 | 0.0051 | 0.0051 |
| C: cd + Dixon-Coles | 1.0000 | 5.0000 | 1140 | 0.9653 | 0.5716 | 0.5447 | 0.0017 | 0.0012 | 0.0075 | 0.0051 |
| cd + promoted, joint | 0.7500 | 5.0000 | 1140 | 0.9641 | 0.5713 | 0.5482 | 0.0006 | 0.0014 | 0.0064 | 0.0054 |
| working: cd + promoted (2-stage) | 1.0000 | 5.0000 | 1140 | 0.9635 | 0.5707 | 0.5482 | 0.0000 | 0.0000 | 0.0058 | 0.0052 |
| working + team home adv (kappa_h=20) | 1.0000 | 5.0000 | 1140 | 0.9634 | 0.5706 | 0.5518 | -0.0001 | 0.0008 | 0.0057 | 0.0053 |
| working + shots (xi_s=0.6) | 1.5000 | 5.0000 | 1140 | 0.9593 | 0.5672 | 0.5491 | -0.0042 | 0.0030 | 0.0016 | 0.0046 |
| working + shots (xi_s=1.0) | 1.5000 | 5.0000 | 1140 | 0.9601 | 0.5677 | 0.5456 | -0.0034 | 0.0030 | 0.0024 | 0.0047 |

## Split by subset

| window | subset | matches | model | ref_log_loss | log_loss | dLL_vs_ref | se |
|---|---|---|---|---|---|---|---|
| early | all matches | 1140 | B: Poisson | 0.9635 | 0.9625 | -0.0011 | 0.0011 |
| early | all matches | 1140 | B + closed doors | 0.9635 | 0.9628 | -0.0008 | 0.0007 |
| early | all matches | 1140 | C: cd + Dixon-Coles | 0.9635 | 0.9653 | 0.0017 | 0.0012 |
| early | all matches | 1140 | cd + promoted, joint | 0.9635 | 0.9641 | 0.0006 | 0.0014 |
| early | all matches | 1140 | working + team home adv (kappa_h=20) | 0.9635 | 0.9634 | -0.0001 | 0.0008 |
| early | all matches | 1140 | working + shots (xi_s=0.6) | 0.9635 | 0.9593 | -0.0042 | 0.0030 |
| early | all matches | 1140 | working + shots (xi_s=1.0) | 0.9635 | 0.9601 | -0.0034 | 0.0030 |
| early | involving a newly promoted team | 324 | B: Poisson | 0.9588 | 0.9608 | 0.0020 | 0.0025 |
| early | involving a newly promoted team | 324 | B + closed doors | 0.9588 | 0.9605 | 0.0017 | 0.0020 |
| early | involving a newly promoted team | 324 | C: cd + Dixon-Coles | 0.9588 | 0.9630 | 0.0042 | 0.0027 |
| early | involving a newly promoted team | 324 | cd + promoted, joint | 0.9588 | 0.9564 | -0.0024 | 0.0040 |
| early | involving a newly promoted team | 324 | working + team home adv (kappa_h=20) | 0.9588 | 0.9604 | 0.0016 | 0.0014 |
| early | involving a newly promoted team | 324 | working + shots (xi_s=0.6) | 0.9588 | 0.9560 | -0.0028 | 0.0066 |
| early | involving a newly promoted team | 324 | working + shots (xi_s=1.0) | 0.9588 | 0.9585 | -0.0003 | 0.0068 |
| early | all other matches | 816 | B: Poisson | 0.9654 | 0.9631 | -0.0023 | 0.0012 |
| early | all other matches | 816 | B + closed doors | 0.9654 | 0.9637 | -0.0017 | 0.0005 |
| early | all other matches | 816 | C: cd + Dixon-Coles | 0.9654 | 0.9661 | 0.0007 | 0.0013 |
| early | all other matches | 816 | cd + promoted, joint | 0.9654 | 0.9672 | 0.0018 | 0.0012 |
| early | all other matches | 816 | working + team home adv (kappa_h=20) | 0.9654 | 0.9646 | -0.0008 | 0.0009 |
| early | all other matches | 816 | working + shots (xi_s=0.6) | 0.9654 | 0.9607 | -0.0047 | 0.0032 |
| early | all other matches | 816 | working + shots (xi_s=1.0) | 0.9654 | 0.9607 | -0.0047 | 0.0033 |

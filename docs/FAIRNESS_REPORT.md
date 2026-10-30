# Fairness Report
Subgroup performance on the holdout test set. Groups below the minimum size or without both classes are shown but not scored. Differences are descriptive; they do not establish discrimination or its absence.

## Disparity summary (max-min spread across groups)
| attribute | groups | auc_spread | recall_spread | fpr_spread | max_abs_calibration_gap |
|---|---|---|---|---|---|
| AGE_BAND | 5 | 0.0451 | 0.3524 | 0.2115 | 0.0078 |
| CODE_GENDER | 2 | 0.0019 | 0.0769 | 0.0437 | 0.01 |
| INCOME_QUARTILE | 4 | 0.0247 | 0.0642 | 0.0428 | 0.0035 |
| NAME_INCOME_TYPE | 4 | 0.0294 | 0.2625 | 0.1067 | 0.0092 |

## Subgroup metrics
| attribute | group | n | positives | observed_rate | mean_predicted_pd | calibration_gap | reportable | roc_auc | recall | false_positive_rate | flagged_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CODE_GENDER | F | 40561 | 2836 | 0.0699 | 0.0742 | 0.0043 | True | 0.778 | 0.4365 | 0.1021 | 0.1255 |
| CODE_GENDER | M | 20940 | 2129 | 0.1017 | 0.0916 | -0.01 | True | 0.7799 | 0.5134 | 0.1458 | 0.1832 |
| CODE_GENDER | XNA | 2 | 0 | 0.0 | 0.0719 | 0.0719 | False | nan | nan | nan | nan |
| AGE_BAND | <25 | 2392 | 280 | 0.1171 | 0.1249 | 0.0078 | True | 0.7425 | 0.6 | 0.2571 | 0.2972 |
| AGE_BAND | 25-34 | 14368 | 1513 | 0.1053 | 0.1053 | 0.0 | True | 0.7805 | 0.5783 | 0.1851 | 0.2265 |
| AGE_BAND | 35-44 | 16889 | 1399 | 0.0828 | 0.0821 | -0.0007 | True | 0.7864 | 0.4789 | 0.1225 | 0.152 |
| AGE_BAND | 45-54 | 14039 | 1030 | 0.0734 | 0.0718 | -0.0016 | True | 0.7862 | 0.4214 | 0.0906 | 0.1148 |
| AGE_BAND | 55+ | 13815 | 743 | 0.0538 | 0.0522 | -0.0015 | True | 0.7413 | 0.2476 | 0.0456 | 0.0565 |
| INCOME_QUARTILE | Q1 low | 20075 | 1642 | 0.0818 | 0.0842 | 0.0025 | True | 0.7819 | 0.4878 | 0.1279 | 0.1573 |
| INCOME_QUARTILE | Q2 | 10678 | 921 | 0.0863 | 0.0853 | -0.0009 | True | 0.7692 | 0.4864 | 0.1314 | 0.162 |
| INCOME_QUARTILE | Q3 | 16356 | 1387 | 0.0848 | 0.0813 | -0.0035 | True | 0.7803 | 0.4701 | 0.1182 | 0.1481 |
| INCOME_QUARTILE | Q4 high | 14394 | 1015 | 0.0705 | 0.0691 | -0.0014 | True | 0.7939 | 0.4236 | 0.0886 | 0.1122 |
| NAME_INCOME_TYPE | Businessman | 1 | 0 | 0.0 | 0.021 | 0.021 | False | nan | nan | nan | nan |
| NAME_INCOME_TYPE | Commercial associate | 14344 | 1012 | 0.0706 | 0.0749 | 0.0043 | True | 0.7764 | 0.4298 | 0.1091 | 0.1317 |
| NAME_INCOME_TYPE | Maternity leave | 2 | 2 | 1.0 | 0.3012 | -0.6988 | False | nan | nan | nan | nan |
| NAME_INCOME_TYPE | Pensioner | 11228 | 617 | 0.055 | 0.053 | -0.002 | True | 0.7535 | 0.2658 | 0.0451 | 0.0573 |
| NAME_INCOME_TYPE | State servant | 4185 | 225 | 0.0538 | 0.063 | 0.0092 | True | 0.7829 | 0.3911 | 0.0788 | 0.0956 |
| NAME_INCOME_TYPE | Student | 7 | 0 | 0.0 | 0.1144 | 0.1144 | False | nan | nan | nan | nan |
| NAME_INCOME_TYPE | Unemployed | 5 | 3 | 0.6 | 0.2206 | -0.3794 | False | nan | nan | nan | nan |
| NAME_INCOME_TYPE | Working | 31731 | 3106 | 0.0979 | 0.0943 | -0.0036 | True | 0.7801 | 0.5283 | 0.1519 | 0.1887 |

# Business Findings
## Portfolio (holdout test set; observed vs predicted shown separately)
- Applicants: 61,503; historical positive labels: 4,965 (observed rate 8.07%)
- Average model-predicted PD: 8.01%; High+Very High share: 14.09%
- Band distribution: LOW 51.7%, MEDIUM 34.2%, HIGH 10.4%, VERY_HIGH 3.6%

## Risk deciles (1 = highest predicted risk)
| decile | n | observed_rate | mean_predicted_pd | positives | lift | cum_capture |
|---|---|---|---|---|---|---|
| 1 | 6151 | 0.2988 | 0.2963 | 1838 | 3.7015 | 0.3702 |
| 2 | 6150 | 0.1559 | 0.1464 | 959 | 1.9316 | 0.5633 |
| 3 | 6150 | 0.1023 | 0.0983 | 629 | 1.2669 | 0.69 |
| 4 | 6150 | 0.0741 | 0.0717 | 456 | 0.9185 | 0.7819 |
| 5 | 6151 | 0.0491 | 0.0545 | 302 | 0.6082 | 0.8427 |
| 6 | 6150 | 0.0431 | 0.0425 | 265 | 0.5338 | 0.8961 |
| 7 | 6150 | 0.0343 | 0.0335 | 211 | 0.425 | 0.9386 |
| 8 | 6150 | 0.0228 | 0.026 | 140 | 0.282 | 0.9668 |
| 9 | 6150 | 0.0182 | 0.0195 | 112 | 0.2256 | 0.9893 |
| 10 | 6151 | 0.0086 | 0.0124 | 53 | 0.1067 | 1.0 |

## Illustrative ECL (assumed LGD scenarios; NOT realised losses, no IFRS 9 claim)
| optimistic | base | conservative |
|---|---|---|
| 684102537.0833 | 1231384566.75 | 1778666596.4167 |  (total EAD 36,765,081,600)

## Top global model drivers (mean |SHAP|, model behaviour not causation)
| feature | mean_abs_shap | mean_shap |
|---|---|---|
| EXT_SOURCE_2 | 0.3002 | -0.0014 |
| EXT_SOURCE_3 | 0.2696 | -0.0047 |
| EXT_SOURCE_1 | 0.1703 | -0.0019 |
| PREV_CNT_PAYMENT_MEAN | 0.1194 | -0.0005 |
| CREDIT_ANNUITY_RATIO | 0.1103 | -0.0015 |
| PREV_CREDIT_APPLICATION_RATIO | 0.0875 | -0.0019 |
| DAYS_EMPLOYED | 0.0839 | -0.0015 |
| LATE_PAYMENT_RATIO | 0.0805 | -0.0006 |
| BUREAU_DEBT_CREDIT_RATIO | 0.0764 | -0.0024 |
| NAME_EDUCATION_TYPE | 0.0733 | 0.0009 |
| AMT_ANNUITY | 0.07 | 0.0003 |
| DAYS_BIRTH | 0.058 | -0.0021 |
| OWN_CAR_AGE | 0.056 | 0.0011 |
| POS_RECORD_COUNT | 0.0486 | -0.0022 |
| PREV_AMT_APPLICATION_MEAN | 0.0464 | -0.0017 |


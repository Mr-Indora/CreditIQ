# Model Card - CreditIQ PD model

**Version:** 1.0.0  |  **Selected model:** lightgbm  |  **Calibration:** none  |  **Seed:** 42

## Intended use
Decision-SUPPORT estimate of Probability of Default (PD) for loan applications, to prioritise human review. Not an autonomous approve/reject system.

## Prohibited use
Automated adverse decisions without human review; use outside the population/time of the training data; use of outputs as evidence of causation; regulatory capital/IFRS 9 reporting.

## Data
Home Credit Default Risk (Kaggle). Label = competition's historical repayment-difficulty flag (not a universal default event).
Splits: fit 184,506 | calibration 61,502 | untouched holdout test 61,503; input features: 179.

## Holdout performance (calibrated, threshold 0.1471 chosen on calibration split)
| Metric | Value |
|---|---|
| ROC-AUC | 0.7823 |
| PR-AUC | 0.2778 |
| KS | 0.4266 |
| Brier | 0.06606 |
| Log loss | 0.23829 |
| Precision / Recall / F1 | 0.261 / 0.469 / 0.336 |
| Specificity | 0.883 |

### Holdout comparison
| model | roc_auc | pr_auc | precision | recall | f1 | ks | brier | log_loss |
|---|---|---|---|---|---|---|---|---|
| dummy | 0.5 | 0.0807 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0742 | 0.2805 |
| logreg | 0.7647 | 0.2472 | 0.2366 | 0.4518 | 0.3106 | 0.3963 | 0.1961 | 0.5773 |
| random_forest | 0.7602 | 0.2405 | 0.2358 | 0.4359 | 0.306 | 0.3898 | 0.1714 | 0.5247 |
| xgboost | 0.7817 | 0.277 | 0.2662 | 0.4506 | 0.3346 | 0.4275 | 0.0661 | 0.2385 |
| lightgbm (selected, raw) | 0.7823 | 0.2778 | 0.2612 | 0.4695 | 0.3356 | 0.4266 | 0.0661 | 0.2383 |
| lightgbm (selected, calibrated=none) | 0.7823 | 0.2778 | 0.2612 | 0.4695 | 0.3356 | 0.4266 | 0.0661 | 0.2383 |

### Cross-validation (development set)
| model | cv_roc_auc_mean | cv_pr_auc_mean | cv_brier_mean | cv_log_loss_mean | cv_roc_auc_std | fit_seconds_per_fold | interpretability | cv_rows |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.7766 | 0.2633 | 0.0668 | 0.241 | 0.0049 | 37.3682 | medium (SHAP) | 120000 |
| xgboost | 0.7757 | 0.2634 | 0.0668 | 0.241 | 0.0041 | 34.4808 | medium (SHAP) | 120000 |
| logreg | 0.7591 | 0.2362 | 0.1979 | 0.5815 | 0.006 | 20.2402 | high (coefficients) | 120000 |
| random_forest | 0.754 | 0.2329 | 0.1663 | 0.5133 | 0.0038 | 134.7801 | medium | 120000 |
| dummy | 0.5 | 0.0807 | 0.0742 | 0.2805 | 0.0 | 10.7464 | nan | 120000 |

## Limitations
See LIMITATIONS.md. Temporal validity could not be independently verified (dataset provides relative day offsets only). Sensitive attribute CODE_GENDER is excluded from model inputs; age is retained and is a governance decision for the deploying institution.

## Human-review requirements
Every recommendation maps to a review level (Further assessment / Manual review / Enhanced review / Escalated review). Risk-band thresholds are illustrative starting values, not industry-approved cutoffs.

# Technical Interview Preparation
- **Why calibrate?** Class weighting/boosting distort probabilities; PD bands and ECL need reliable probabilities. Compare Brier/log loss/calibration curve; fit calibrator on a split disjoint from the holdout.
- **Leakage handling?** Day-offset sign audit, neutralised future values, TARGET excluded, preprocessing fitted on train only; honest statement that temporal validity cannot be proven here.
- **Why PR-AUC/KS over accuracy?** ~8% positive rate (check your run) makes accuracy uninformative.
- **Why SHAP is not causal?** It attributes model output given correlations in data.
- **ECL caveats?** Assumed LGD scenarios; not IFRS 9; not realised loss.
- **Fairness?** Descriptive subgroup metrics, sensitive attribute excluded from inputs, limits of such audits.
- **Scaling?** DuckDB streaming SQL, Parquet, CV subsample config, NaN-native boosters.
- **What would you do next?** Out-of-time validation, monotonic constraints, scorecard benchmark, champion-challenger, proper feature store.

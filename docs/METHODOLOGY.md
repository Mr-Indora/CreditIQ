# Methodology
1. **Audit**: row/column/missing/duplicate/key checks per table, referential integrity, data dictionary (official descriptions merged if `HomeCredit_columns_description.csv` is present).
2. **Leakage/temporal**: sign audit of every DAYS_*/MONTHS_BALANCE column; unsafe values neutralised; features documented with source and availability. Questionable features would be excluded from the primary model.
3. **EDA**: figures with what/why/observation/interpretation/limitation; observations computed from data.
4. **Statistics**: Mann-Whitney U + Cliff's delta (+ bootstrap CI of median difference) for numeric features; chi-square + Cramer's V + Wilson CI for categoricals; Holm and Benjamini-Hochberg corrections; practical-significance labels. Association, not causation.
5. **Modelling**: baseline-first (Dummy), then LogReg, RandomForest, XGBoost, LightGBM; stratified CV ROC-AUC/PR-AUC/Brier/log-loss/time; class weighting experiment; optional Optuna tuning (off by default).
6. **Calibration**: none/Platt/isotonic compared by CV Brier on the calibration split; threshold chosen by F1 on that split; everything reported on the holdout.
7. **Explainability**: SHAP on the fitted model's transformed features (log-odds of the uncalibrated model).
8. **Risk engine / ECL / portfolio / fairness / monitoring**: see module docstrings and `configs/config.yaml` (bands, LGD, PSI thresholds are assumptions).

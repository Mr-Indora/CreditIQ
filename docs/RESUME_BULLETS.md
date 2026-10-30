# Resume bullets (evidence-backed; independent project)
- Built CreditIQ, an end-to-end credit-risk platform on the Home Credit dataset: DuckDB SQL aggregation of 7 relational tables into an applicant-level feature table with a documented leakage/temporal audit.
- Trained and compared 5 models with stratified CV and an untouched holdout; selected lightgbm with none calibration - holdout ROC-AUC 0.782, PR-AUC 0.278, KS 0.427, Brier 0.0661 (n=61,503).
- Delivered SHAP explanations, an illustrative ECL scenario engine, fairness/drift reports, a FastAPI prediction service and Streamlit dashboard, with pytest-covered modules.

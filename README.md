# CreditIQ - Credit Risk Intelligence & Decision Support Platform

End-to-end applied data-science project on the **Home Credit Default Risk** dataset: relational feature engineering (DuckDB SQL) -> leakage audit -> EDA & statistics -> model comparison -> calibration -> SHAP -> risk engine -> illustrative ECL -> portfolio analytics -> fairness & drift monitoring -> FastAPI + Streamlit app. Built **Colab-first** (Google Drive for persistence, GitHub for source).

> **Status honesty:** all code is implemented and the full pipeline + 36 pytest tests were executed on *synthetic data in the Home Credit schema* (smoke test of the code). **No real-data metrics exist until you run the notebooks on the real dataset.** Docs that quote metrics (MODEL_CARD, BUSINESS_FINDINGS, RESUME_BULLETS ...) are generated from executed artifacts and are labelled if produced from synthetic data.

## Business problem
Estimate the Probability of Default (PD) at application time, segment applicants into configurable risk bands, explain each score, quantify illustrative expected loss, and support (never replace) human lending review.

## Dataset
Kaggle `home-credit-default-risk` (8 tables). `TARGET=1` is the competition's historical repayment-difficulty label, not a universal default event. `application_test` has no labels and is never used for evaluation.

## Architecture
```
Kaggle/CSV -> Parquet (DuckDB) -> SQL aggregation (sql/*.sql) -> feature table (1 row / SK_ID_CURR)
  -> split: holdout test | development = (fit | calibration)
  -> CV model comparison -> selected model -> calibration -> thresholds -> bundle (joblib + manifest)
  -> RiskEngine (validate -> PD -> band -> SHAP -> ECL) -> FastAPI / Streamlit
  -> portfolio, fairness, drift reports -> auto-generated docs -> export zip
```
Shared logic lives in `src/creditiq/`; notebooks are thin runners.

## Technology
Python, DuckDB, pandas, scikit-learn, XGBoost, LightGBM, SHAP, SciPy, Plotly/Matplotlib/Seaborn, FastAPI/Pydantic, Streamlit, pytest, GitHub Actions.

## Quick start (Google Colab)
1. Put `CreditIQ.zip` in `MyDrive/` (the notebook unzips it to `MyDrive/CreditIQ/`).
2. Add Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` (and accept the competition rules on Kaggle) **or** upload the 8 CSVs to `MyDrive/CreditIQ/data/raw/`.
3. Open **`notebooks/00_COLAB_MASTER_RUN.ipynb`** and *Run all*. It runs every stage, prints a sanity verdict on the results, runs pytest, generates the docs from your real run, launches API + dashboard through Colab's private port proxy and exports a portable bundle. Finished stages are skipped on re-run (checkpoints live on Drive).
4. Optional: set `PROFILE = "max"` and run again for the full-data, Optuna-tuned model (`configs/config_max.yaml`).

Prefer step-by-step? Notebooks `01` -> `06` run the same stages one at a time. Full guide, time estimates, expected metric ranges and troubleshooting: **`docs/COLAB_RUNBOOK.md`**.

CLI equivalent: `python -m creditiq.pipeline --stage all` (add `--synthetic` for a smoke test only).

## Method highlights
- Leakage: history aggregates use only records at or before application time per dataset day-offset convention; future-dated values are neutralised (`DAYS_ENDDATE_FACT > 0`), `DAYS_EMPLOYED=365243` sentinel handled; `TARGET` never a feature; `CODE_GENDER` excluded from model inputs (kept for fairness audit only). The audit checks consistency - it cannot prove temporal validity.
- Imbalance: class weights evaluated; SMOTE off by default and, if enabled, runs only inside the training pipeline.
- Evaluation: ROC-AUC, PR-AUC, KS, Brier, log loss, precision/recall/F1/specificity, deciles, lift; accuracy is not a selection metric.
- Calibration (Platt/isotonic) and decision threshold are fitted on a calibration split, never on the holdout.
- ECL = PD x LGD x EAD with assumed LGD scenarios (25/45/65%). Illustrative only; no IFRS 9 claim.

## Limitations
See `docs/LIMITATIONS.md`. Dashboard screenshots should be added after your real run.

## Layout
`notebooks/ src/creditiq/ sql/ api/ dashboard/ configs/ tests/ docs/ data/ artifacts/ exports/ logs/ tools/`

# CreditIQ - Colab Runbook

Operational guide for running CreditIQ end to end on Google Colab with the real Home Credit data.
Everything below is executed from one notebook: `notebooks/00_COLAB_MASTER_RUN.ipynb`.

> **Status honesty.** The code and the 36 tests were verified on synthetic data in the Home Credit schema
> (and the master notebook was executed end to end on it). No real-data metric exists until *you* run this on
> the real dataset. Anything quoted as an expected range below comes from what is commonly reported for this
> public dataset, not from a run of this repo.

---

## 1. One-time setup (10 minutes)

| Step | Action |
|---|---|
| 1 | Put `CreditIQ.zip` in the root of Google Drive (`MyDrive/`). The notebook unzips it to `MyDrive/CreditIQ/`. If the zip is missing it asks you to upload it. |
| 2 | Kaggle: open the competition page for *Home Credit Default Risk* and click **Join / accept rules** (without this the API download returns 403). |
| 3 | Kaggle: Settings -> API -> create a token. In Colab (left sidebar, key icon) add Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`, enable **Notebook access** for both. If your Kaggle account only issues the new-style access token, add it as secret `KAGGLE_API_TOKEN` instead. |
| 4 | **Fallback (no API):** download the competition zip in the browser, upload the 8 CSVs to `MyDrive/CreditIQ/data/raw/`. The pipeline detects them automatically. |
| 5 | Runtime -> Change runtime type -> **CPU** is enough (LightGBM/XGBoost run on CPU here). High-RAM, if offered, only makes it safer. |

Drive space needed: about 3 GB raw CSVs + about 1 GB Parquet + small artifacts. Free Drive (15 GB) is enough.

## 2. Run

1. Open `notebooks/00_COLAB_MASTER_RUN.ipynb` from Drive in Colab.
2. Leave `PROFILE = "fast"` for the first run. Run all cells.
3. Review section 4 of the notebook (results + sanity verdict).
4. Optional second pass: set `PROFILE = "max"` and run again (see section 4 below).

If Colab disconnects, re-open and run all again. Every stage writes a checkpoint to `logs/pipeline_state.json`, SQL aggregates and CV results are cached on Drive, so finished work is skipped.

### Stages

| Stage | What happens | Main outputs |
|---|---|---|
| ingest | Kaggle download or manual CSVs -> schema validation -> CSV to Parquet (DuckDB) | `data/interim/*.parquet`, `file_validation.csv` |
| audit | dataset audit, key relationships, data dictionary, leakage audit, SQL analysis queries | `artifacts/reports/*audit*.csv`, `sql_*.csv` |
| aggregate | 5 SQL files aggregate history tables to one row per applicant | `data/interim/agg_*.parquet` |
| features | joins + ratios, train/test consistency assertions | `data/processed/features_*.parquet` |
| eda / stats | plots, hypothesis tests, correlations | `eda_report.md`, `statistical_tests.csv` |
| train | CV comparison -> selected model -> calibration -> threshold -> holdout evaluation | `metrics.json`, `cv_comparison.csv`, `creditiq_bundle.joblib` |
| analytics | SHAP, risk bands, ECL, portfolio, fairness, drift | `shap_global_importance.csv`, `fairness_report.csv`, `drift_report.csv` ... |
| report | docs regenerated from executed artifacts | `docs/MODEL_CARD.md`, `BUSINESS_FINDINGS.md`, `RESUME_BULLETS.md` ... |
| export | portable zip + manifest with SHA-256 checksums | `exports/creditiq_export_*.zip` |

### Time estimates (not measured on real data - treat as rough)

| Profile | Free Colab CPU (2 vCPU) | Notes |
|---|---|---|
| fast | roughly 30-60 min | Slowest parts: CSV->Parquet + SQL aggregation, then Random Forest and XGBoost CV. |
| max | roughly 1-2 h extra | Whole development set in CV + 30 Optuna trials. Free sessions can idle-disconnect; keep the tab open. |

## 3. What results to expect (and what they mean)

Home Credit is a hard, noisy dataset. Commonly reported holdout ROC-AUC levels:

| Setup | Typical ROC-AUC |
|---|---|
| Application table only, gradient boosting | about 0.74 - 0.76 |
| + bureau / previous application / instalments / card aggregates (this project) | about 0.76 - 0.79 |
| Top Kaggle leaderboard (heavy stacking, hundreds of features) | about 0.80 |

The notebook prints a verdict after training:

- **0.70 - 0.85:** plausible. Report it as is.
- **> 0.85:** suspicious. On this dataset that almost always means leakage, not a great model. Open `leakage_audit.csv` and `feature_availability.csv` before believing it.
- **< 0.70:** below the usual range. Check `file_validation.csv`, row counts in `dataset_audit.csv`, and that all 5 aggregates were built.

What makes the project strong is not a record AUC. It is the things most notebooks skip: leakage audit with time-offset rules, calibration fitted on a separate split, threshold chosen without touching the holdout, honest ECL labelling, fairness audit with a sensitive attribute excluded from the model, drift monitoring, tests, and generated documentation.

## 4. Improving the score (in order of payoff)

1. **`PROFILE = "max"`** - the full development set in CV plus Optuna-tuned LightGBM (learning rate, leaves, regularisation, number of trees). Usually the cheapest gain. Previous metrics are saved to `artifacts/runs/fast/` so you can show a before/after table.
2. **More history features** - e.g. ratios of recent vs old instalment behaviour, last-N-months windows on `credit_card_balance` and `POS_CASH_balance` (add the SQL in `sql/`, document the formula in `FEATURE_DOCS` in `features.py`, keep the day-offset <= 0 rule).
3. **Seed averaging / blending** of LightGBM and XGBoost - small but reliable gain; keep calibration on the calibration split.

Do not tune on the holdout. If you change features or models, re-run; the holdout must stay untouched.

## 5. After the run: what to do with the outputs

| Item | Where | Use |
|---|---|---|
| Docs with real numbers | `docs/MODEL_CARD.md`, `BUSINESS_FINDINGS.md`, `FAIRNESS_REPORT.md` | GitHub README / portfolio |
| Resume bullets | `docs/RESUME_BULLETS.md` | Generated only from a real (non-synthetic) run |
| Figures | `artifacts/reports/figures/` | README, slides |
| Dashboard screenshots | notebook section 8 | README (take them while the app is up) |
| Bundle | `exports/creditiq_export_*.zip` | Backup / hand-off; manifest verifies checksums |
| GitHub | commented cell in `06_application_testing_export.ipynb` | Token from Colab Secrets, never hardcoded |

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Kaggle 403 / "Dataset unavailable" | Competition rules not accepted, or secrets not shared with the notebook | Accept rules; re-check secret **Notebook access** toggle; or use manual upload |
| `Dataset validation failed` | Missing CSV or wrong folder | The table shows which file/column is missing; all 8 CSVs must be in `data/raw/` (or one sub-folder) |
| Session crashed (RAM) | Another heavy session, or Parquet conversion on a small runtime | Restart session, run all again (checkpoints resume); switch to High-RAM if available |
| Import error after `pip install` | Colab preinstalled package got upgraded | Runtime -> Restart session, run all again |
| Training too slow | RF/XGBoost CV on free CPU | In `configs/config.yaml` lower `cv.sample_n` (e.g. 60000) or remove `random_forest` from `models` |
| Want to recompute everything | - | Set `FORCE = True` in notebook settings (or delete `logs/pipeline_state.json`) |
| Switched profile but results look old | - | Switching profile retrains automatically; if in doubt set `FORCE = True` |
| Dashboard window blank | Streamlit still starting | Wait 20 s, re-run the launch cell |
| Synthetic banner in docs | `SYNTHETIC_SMOKE` was used | Delete `data/raw/*`, `data/interim/*`, `logs/pipeline_state.json`; run again with real data |

## 7. Local / CLI equivalent

```bash
pip install -r requirements.txt
export CREDITIQ_ROOT=$PWD PYTHONPATH=src
python -m creditiq.pipeline --stage all                     # real data in data/raw/
CREDITIQ_CONFIG=config_max.yaml python -m creditiq.pipeline --stage train --force
python -m creditiq.pipeline --stage all --synthetic         # smoke test only
pytest -q
```

## 8. Configuration reference

| File | Purpose |
|---|---|
| `configs/config.yaml` | Fast baseline: CV on a 120k stratified sample, no tuning |
| `configs/config_max.yaml` | Max quality: full development set, Random Forest dropped, 30 Optuna trials |
| env `CREDITIQ_CONFIG` | Name of the config file in `configs/` to use (set automatically by the master notebook) |
| env `CREDITIQ_ROOT` | Project root (set automatically) |

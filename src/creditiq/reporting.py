"""Auto-generated docs from EXECUTED artifacts only. Missing artifacts are reported as 'not run yet' - never invented."""
from __future__ import annotations
import pandas as pd
from .config import read_json, read_state


def _md(df: pd.DataFrame, floatfmt=4) -> str:
    if df is None or df.empty: return "_n/a_"
    d = df.copy()
    for c in d.select_dtypes("float").columns: d[c] = d[c].round(floatfmt)
    cols = list(d.columns); rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def _read_csv(p): return pd.read_csv(p) if p.exists() else None
def _read_json(p): return read_json(p) if p.exists() else None


def _is_synthetic(paths) -> bool:
    return bool(read_state(paths)["completed"].get("ingest", {}).get("synthetic", False))


def generate_reports(paths, cfg):
    m = _read_json(paths.metrics / "metrics.json"); syn = _is_synthetic(paths)
    banner = ("> **WARNING: this run used SYNTHETIC smoke-test data. Numbers below are meaningless; do not cite them.**\n\n" if syn else "")
    d = paths.docs
    # MODEL CARD
    if m:
        t = m["test_calibrated"]; hc = _read_csv(paths.metrics / "holdout_comparison.csv"); cv = _read_csv(paths.metrics / "cv_comparison.csv")
        (d / "MODEL_CARD.md").write_text(f"""# Model Card - CreditIQ PD model
{banner}
**Version:** {m['model_version']}  |  **Selected model:** {m['selected_model']}  |  **Calibration:** {m['calibration_method']}  |  **Seed:** {m['seed']}

## Intended use
Decision-SUPPORT estimate of Probability of Default (PD) for loan applications, to prioritise human review. Not an autonomous approve/reject system.

## Prohibited use
Automated adverse decisions without human review; use outside the population/time of the training data; use of outputs as evidence of causation; regulatory capital/IFRS 9 reporting.

## Data
Home Credit Default Risk (Kaggle). Label = competition's historical repayment-difficulty flag (not a universal default event).
Splits: fit {m['split_sizes']['fit']:,} | calibration {m['split_sizes']['calibration']:,} | untouched holdout test {m['split_sizes']['test']:,}; input features: {m['split_sizes']['features_in']}.

## Holdout performance (calibrated, threshold {m['decision_threshold']:.4f} chosen on calibration split)
| Metric | Value |
|---|---|
| ROC-AUC | {t['roc_auc']:.4f} |
| PR-AUC | {t['pr_auc']:.4f} |
| KS | {t['ks']:.4f} |
| Brier | {t['brier']:.5f} |
| Log loss | {t['log_loss']:.5f} |
| Precision / Recall / F1 | {t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f} |
| Specificity | {t['specificity']:.3f} |

### Holdout comparison
{_md(hc)}

### Cross-validation (development set)
{_md(cv)}

## Limitations
See LIMITATIONS.md. Temporal validity could not be independently verified (dataset provides relative day offsets only). Sensitive attribute CODE_GENDER is excluded from model inputs; age is retained and is a governance decision for the deploying institution.

## Human-review requirements
Every recommendation maps to a review level (Further assessment / Manual review / Enhanced review / Escalated review). Risk-band thresholds are illustrative starting values, not industry-approved cutoffs.
""")
    else: (d / "MODEL_CARD.md").write_text("# Model Card\n\nNot generated: training stage has not been executed yet.\n")
    # FAIRNESS
    fr = _read_csv(paths.reports / "fairness_report.csv"); ds = _read_csv(paths.reports / "fairness_disparity.csv")
    (d / "FAIRNESS_REPORT.md").write_text(f"# Fairness Report\n{banner}" + ("Subgroup performance on the holdout test set. Groups below the minimum size or without both classes are shown but not scored. Differences are descriptive; they do not establish discrimination or its absence.\n\n## Disparity summary (max-min spread across groups)\n" + _md(ds) + "\n\n## Subgroup metrics\n" + _md(fr) if fr is not None else "Not run yet.") + "\n")
    # BUSINESS FINDINGS
    ps = _read_json(paths.metrics / "portfolio_summary.json"); ec = _read_json(paths.metrics / "ecl_summary.json"); sh = _read_csv(paths.metrics / "shap_global_importance.csv")
    dec = _read_csv(paths.reports / "portfolio_risk_deciles.csv")
    body = "Not run yet."
    if ps:
        body = f"""## Portfolio (holdout test set; observed vs predicted shown separately)
- Applicants: {ps['total_applicants']:,}; historical positive labels: {ps['historical_positive_label_count']:,} (observed rate {ps['historical_target_rate_observed']:.2%})
- Average model-predicted PD: {ps['average_predicted_pd_model_estimate']:.2%}; High+Very High share: {ps['high_risk_population_pct']:.2%}
- Band distribution: {', '.join(f'{k} {v:.1%}' for k, v in ps['risk_segment_distribution'].items())}

## Risk deciles (1 = highest predicted risk)
{_md(dec)}

## Illustrative ECL (assumed LGD scenarios; NOT realised losses, no IFRS 9 claim)
{_md(pd.DataFrame([ec['scenario_total']]))}  (total EAD {ec['total_ead']:,.0f})

## Top global model drivers (mean |SHAP|, model behaviour not causation)
{_md(sh.head(15)) if sh is not None else 'n/a'}
"""
    (d / "BUSINESS_FINDINGS.md").write_text(f"# Business Findings\n{banner}{body}\n")
    # DATA DICTIONARY
    dd = _read_csv(paths.reports / "data_dictionary.csv")
    (d / "DATA_DICTIONARY.md").write_text("# Data Dictionary (generated from loaded data)\n\n" + (_md(dd[["table", "column_name", "column_type", "missing_pct", "description"]].fillna("")) if dd is not None else "Not run yet.") + "\n")
    # STATUS
    st = read_state(paths)
    (d / "PROJECT_STATUS.md").write_text("# Project Status (from pipeline_state.json)\n\n" + banner + _md(pd.DataFrame([{"stage": k, **{a: b for a, b in v.items()}} for k, v in st["completed"].items()])) + f"\n\nLast completed: {st.get('last_completed', 'none')}\n")
    # RESUME BULLETS - only from executed, non-synthetic metrics
    if m and not syn:
        t = m["test_calibrated"]; sz = m["split_sizes"]
        (d / "RESUME_BULLETS.md").write_text(f"""# Resume bullets (evidence-backed; independent project)
- Built CreditIQ, an end-to-end credit-risk platform on the Home Credit dataset: DuckDB SQL aggregation of 7 relational tables into an applicant-level feature table with a documented leakage/temporal audit.
- Trained and compared {len(_read_csv(paths.metrics / 'cv_comparison.csv'))} models with stratified CV and an untouched holdout; selected {m['selected_model']} with {m['calibration_method']} calibration - holdout ROC-AUC {t['roc_auc']:.3f}, PR-AUC {t['pr_auc']:.3f}, KS {t['ks']:.3f}, Brier {t['brier']:.4f} (n={sz['test']:,}).
- Delivered SHAP explanations, an illustrative ECL scenario engine, fairness/drift reports, a FastAPI prediction service and Streamlit dashboard, with pytest-covered modules.
""")
    else:
        (d / "RESUME_BULLETS.md").write_text("# Resume bullets\n\nNot generated: requires a completed run on the REAL Home Credit data (metrics must be executed, not synthetic).\n")

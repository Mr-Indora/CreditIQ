"""Portfolio analytics. Observed target rates and model-predicted PDs are reported in separate columns."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .evaluation import decile_table
from .risk_engine import BAND_ORDER, assign_band


def add_bands(scored: pd.DataFrame, bands: dict) -> pd.DataFrame:
    s = scored.copy(); s["RISK_BAND"] = [assign_band(float(x), bands) for x in s["PD"]]; return s


def summary(scored: pd.DataFrame, bands: dict, metrics: dict | None = None) -> dict:
    s = add_bands(scored, bands)
    dist = s.RISK_BAND.value_counts(normalize=True).reindex(BAND_ORDER).fillna(0).to_dict()
    return {"total_applicants": int(len(s)), "historical_positive_label_count": int(s.TARGET.sum()),
            "historical_target_rate_observed": float(s.TARGET.mean()), "average_predicted_pd_model_estimate": float(s.PD.mean()),
            "high_risk_population_pct": float(dist["HIGH"] + dist["VERY_HIGH"]), "risk_segment_distribution": dist,
            "model_performance": None if metrics is None else {k: metrics["test_calibrated"][k] for k in ("roc_auc", "pr_auc", "ks", "brier")},
            "note": "Computed on the holdout test set; observed rates and predicted PDs are distinct quantities."}


def segment_table(scored: pd.DataFrame, col: str, bins=None, labels=None, q=None) -> pd.DataFrame:
    s = scored.copy()
    if q: s["_seg"] = pd.qcut(s[col], q, duplicates="drop")
    elif bins is not None: s["_seg"] = pd.cut(s[col], bins=bins, labels=labels)
    else: s["_seg"] = s[col].astype(object).where(s[col].notna(), "MISSING")
    g = s.groupby("_seg", observed=True).agg(n=("TARGET", "size"), observed_target_rate=("TARGET", "mean"), mean_predicted_pd=("PD", "mean")).reset_index()
    g["_seg"] = g["_seg"].astype(str); return g.rename(columns={"_seg": col})


def build_segment_reports(scored: pd.DataFrame, bands: dict) -> dict[str, pd.DataFrame]:
    s = add_bands(scored, bands); out = {"by_risk_band": segment_table(s, "RISK_BAND")}
    if "AMT_INCOME_TOTAL" in s: out["by_income_quintile"] = segment_table(s, "AMT_INCOME_TOTAL", q=5)
    if "AMT_CREDIT" in s: out["by_credit_amount_quintile"] = segment_table(s, "AMT_CREDIT", q=5)
    if "EMPLOYMENT_DURATION_DAYS" in s: out["by_employment_duration_quintile"] = segment_table(s, "EMPLOYMENT_DURATION_DAYS", q=5)
    if "NAME_INCOME_TYPE" in s: out["by_income_type"] = segment_table(s, "NAME_INCOME_TYPE")
    if "LATE_PAYMENT_RATIO" in s and s.LATE_PAYMENT_RATIO.notna().sum() > 50:
        out["by_previous_payment_behaviour"] = segment_table(s, "LATE_PAYMENT_RATIO", bins=[-0.001, 0, .1, .3, 1.0], labels=["never late", "0-10% late", "10-30% late", ">30% late"])
    out["risk_deciles"] = decile_table(s.TARGET, s.PD)
    return out

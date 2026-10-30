"""Subgroup performance audit. Sensitive attributes are used ONLY for auditing, not as model inputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_metrics(s: pd.DataFrame, group_col: str, thr: float, min_n: int) -> pd.DataFrame:
    rows = []
    for g, d in s.groupby(group_col, observed=True):
        n = len(d); pos = int(d.TARGET.sum())
        pred = d.PD >= thr; neg = (d.TARGET == 0).sum()
        r = {"attribute": group_col, "group": str(g), "n": n, "positives": pos, "observed_rate": d.TARGET.mean(), "mean_predicted_pd": d.PD.mean(),
             "calibration_gap": d.PD.mean() - d.TARGET.mean(), "reportable": n >= min_n and 0 < pos < n}
        if r["reportable"]:
            r.update(roc_auc=roc_auc_score(d.TARGET, d.PD), recall=float((pred & (d.TARGET == 1)).sum() / pos),
                     false_positive_rate=float((pred & (d.TARGET == 0)).sum() / max(1, neg)), flagged_rate=float(pred.mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def fairness_report(scored: pd.DataFrame, thr: float, min_n: int = 300) -> pd.DataFrame:
    s = scored.copy(); parts = []
    if "AGE_YEARS" in s: s["AGE_BAND"] = pd.cut(s.AGE_YEARS, [0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"])
    if "AMT_INCOME_TOTAL" in s: s["INCOME_QUARTILE"] = pd.qcut(s.AMT_INCOME_TOTAL, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
    for c in ["CODE_GENDER", "AGE_BAND", "INCOME_QUARTILE", "NAME_INCOME_TYPE"]:
        if c in s: parts.append(group_metrics(s, c, thr, min_n))
    rep = pd.concat(parts, ignore_index=True)
    ok = rep[rep.reportable]
    # disparity summary: max-min spread of key metrics within each attribute
    return rep


def disparity_summary(rep: pd.DataFrame) -> pd.DataFrame:
    ok = rep[rep.reportable]
    return ok.groupby("attribute").agg(groups=("group", "count"), auc_spread=("roc_auc", lambda x: x.max() - x.min()),
                                       recall_spread=("recall", lambda x: x.max() - x.min()),
                                       fpr_spread=("false_positive_rate", lambda x: x.max() - x.min()),
                                       max_abs_calibration_gap=("calibration_gap", lambda x: x.abs().max())).reset_index()

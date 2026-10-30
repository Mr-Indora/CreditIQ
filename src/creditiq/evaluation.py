"""Metrics, tables and plots. Only executed metrics are reported."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p); return float(np.max(tpr - fpr))


def all_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "specificity": float(tn / max(1, tn + fp)),
            "ks": ks_stat(y, p), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "threshold": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n": int(len(y)), "positive_rate": float(y.mean()), "mean_pred": float(p.mean())}


def threshold_table(y, p, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.unique(np.round(np.quantile(p, np.linspace(0.5, 0.99, 25)), 4))
    return pd.DataFrame([{k: v for k, v in all_metrics(y, p, t).items() if k in ("threshold", "precision", "recall", "f1", "specificity", "fp", "fn")} for t in thresholds])


def best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.nanargmax(f1))])


def decile_table(y, p, n=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), n, labels=range(1, n + 1)).astype(int)  # 1 = highest risk
    t = d.groupby("decile").agg(n=("y", "size"), observed_rate=("y", "mean"), mean_predicted_pd=("p", "mean"), positives=("y", "sum")).reset_index()
    base = d.y.mean(); t["lift"] = t.observed_rate / base
    t["cum_capture"] = t.positives.cumsum() / max(1, t.positives.sum())
    return t


def error_analysis(X: pd.DataFrame, y, p, thr, cols=None) -> pd.DataFrame:
    """Compare feature means of FP vs TN and FN vs TP (descriptive, not causal)."""
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    cols = cols or [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO",
                                "ANNUITY_INCOME_RATIO", "AGE_YEARS", "EMPLOYMENT_DURATION_DAYS", "LATE_PAYMENT_RATIO",
                                "PREVIOUS_REJECTION_COUNT", "BUREAU_PREV_CREDIT_COUNT"] if c in X.columns]
    grp = {"FP": (pred == 1) & (y == 0), "TN": (pred == 0) & (y == 0), "FN": (pred == 0) & (y == 1), "TP": (pred == 1) & (y == 1)}
    rows = [{"group": g, "n": int(m.sum()), **{c: float(pd.to_numeric(X.loc[m, c], errors="coerce").mean()) if m.any() else np.nan for c in cols}} for g, m in grp.items()]
    return pd.DataFrame(rows)


def plot_roc_pr(curves: dict, out_roc, out_pr):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        f, t, _ = roc_curve(y, p); ax.plot(f, t, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - holdout test"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_roc, dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall - holdout test"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_pr, dpi=130); plt.close(fig)


def plot_calibration(curves: dict, out, bins=10):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        fr, mp = calibration_curve(y, p, n_bins=bins, strategy="quantile"); ax.plot(mp, fr, "o-", label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, lim], [0, lim], "k--", lw=.8)
    ax.set(xlabel="Mean predicted PD", ylabel="Observed event rate", title="Calibration (quantile bins) - holdout test"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_confusion(m: dict, out):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"], title=f"Confusion @ thr={m['threshold']:.3f}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

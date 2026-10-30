"""Probability calibration (Platt / isotonic). Fitted on a calibration split, never on the final holdout."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "none"):
        self.method = method; self.model = None

    def fit(self, p, y):
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "platt": return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic": return self.model.predict(p)
        return p


def select_calibrator(p_cal, y_cal, seed=42):
    """Cross-validated comparison on the calibration split; returns (fitted Calibrator, comparison table rows)."""
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal); rows = []
    for m in ["none", "platt", "isotonic"]:
        oof = np.zeros(len(p_cal))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(p_cal, y_cal):
            oof[va] = Calibrator(m).fit(p_cal[tr], y_cal[tr]).predict(p_cal[va])
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        rows.append({"method": m, "cv_brier": brier_score_loss(y_cal, oof), "cv_log_loss": log_loss(y_cal, oof)})
    best = min(rows, key=lambda r: r["cv_brier"])["method"]
    return Calibrator(best).fit(p_cal, y_cal), rows

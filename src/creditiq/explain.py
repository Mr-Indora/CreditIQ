"""SHAP explainability for the actual selected model and feature representation.
Attributions are in log-odds space of the UNCALIBRATED model and describe model behaviour, not causal effects."""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pos_class(sv):
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


class Explainer:
    def __init__(self, pipeline, background_raw: pd.DataFrame, model_name: str, n_background: int = 200, seed: int = 42):
        self.prep = pipeline.named_steps["prep"]; self.clf = pipeline.named_steps["clf"]; self.name = model_name
        bg = self.prep.transform(background_raw.sample(min(len(background_raw), n_background), random_state=seed))
        if model_name == "logreg":
            self.exp = shap.LinearExplainer(self.clf, bg); ev = self.exp.expected_value
        else:
            self.exp = shap.TreeExplainer(self.clf); ev = self.exp.expected_value
        ev = np.asarray(ev).ravel(); self.base_value = float(ev[-1] if len(ev) > 1 else ev[0])

    def shap_values(self, raw: pd.DataFrame):
        Xt = self.prep.transform(raw)
        return Xt, _pos_class(self.exp.shap_values(Xt))

    def local(self, raw: pd.DataFrame, top_k: int = 5) -> list[dict]:
        Xt, sv = self.shap_values(raw); out = []
        for i in range(len(Xt)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            factors = []
            for j in order:
                f = Xt.columns[j]; v = Xt.iloc[i, j]
                factors.append({"feature": f, "value": None if pd.isna(v) else float(v), "contribution_log_odds": float(sv[i, j]),
                                "direction": "increases estimated risk" if sv[i, j] > 0 else "decreases estimated risk"})
            out.append({"baseline_log_odds": self.base_value, "top_factors": factors, "text": human_text(factors)})
        return out

    def global_importance(self, raw: pd.DataFrame) -> pd.DataFrame:
        Xt, sv = self.shap_values(raw)
        return pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(0), "mean_shap": sv.mean(0)}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def summary_plot(self, raw: pd.DataFrame, out, max_display=20):
        Xt, sv = self.shap_values(raw)
        plt.figure(); shap.summary_plot(sv, Xt, max_display=max_display, show=False); plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()


def human_text(factors: list[dict]) -> str:
    parts = [f"{f['feature']} ({'+' if f['contribution_log_odds'] > 0 else '-'}) {f['direction']}" for f in factors[:3]]
    return "Main model drivers: " + "; ".join(parts) + ". Model attribution only - not evidence of causation."

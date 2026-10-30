"""Applicant scoring engine: input validation -> PD -> risk band -> explanation -> ECL. Decision SUPPORT only."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import artifacts as art
from .config import get_paths
from .ecl import ecl_scenarios
from .explain import Explainer
from .features import add_application_features, clean_for_model

BAND_ORDER = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


def assign_band(pd_value: float, bands: dict) -> str:
    for name in BAND_ORDER:
        lo, hi = bands[name]
        if lo <= pd_value < hi: return name
    return "VERY_HIGH"


class RiskEngine:
    def __init__(self, bundle_path: str | Path | None = None, paths=None, background: pd.DataFrame | None = None):
        self.paths = paths or get_paths()
        self.bundle = art.load_bundle(bundle_path or self.paths.models / art.BUNDLE_NAME)
        self.pipe = self.bundle["pipeline"]; self.cal = self.bundle["calibrator"]; self.cols = self.bundle["raw_columns"]
        self.schema = self.bundle["schema"]; self.bands = dict(self.bundle["risk_bands"]); self.version = self.bundle["model_version"]
        self.recs = self.bundle["recommendations"]; self.lgd = self.bundle["lgd_scenarios"]
        self._background = background; self._explainer = None

    # ---- configurable thresholds
    def set_bands(self, bands: dict):
        prev = 0.0
        for n in BAND_ORDER:
            lo, hi = bands[n]
            if not (lo >= prev - 1e-12 and hi > lo): raise ValueError(f"Invalid band {n}: {lo}-{hi}")
            prev = hi
        self.bands = {k: list(v) for k, v in bands.items()}

    # ---- validation
    def validate(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        errors, warnings = [], []
        unknown = [c for c in df.columns if c not in self.schema and c not in ("SK_ID_CURR", "TARGET")]
        base_ok = set(self.schema) | {"DAYS_EMPLOYED", "DAYS_BIRTH", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "CNT_FAM_MEMBERS", "AMT_GOODS_PRICE", "CODE_GENDER"}
        unknown = [c for c in unknown if c not in base_ok]
        if unknown: warnings.append(f"Ignored unknown fields: {unknown[:10]}")
        for c in df.columns:
            sc = self.schema.get(c)
            if not sc: continue
            if sc["type"] == "numeric":
                v = pd.to_numeric(df[c], errors="coerce")
                if (v.isna() & df[c].notna()).any(): errors.append(f"{c}: non-numeric value")
                elif sc["min"] is not None and ((v < sc["min"]) | (v > sc["max"])).any(): warnings.append(f"{c}: outside training range [{sc['min']:.3g}, {sc['max']:.3g}]")
            else:
                bad = set(df[c].dropna().astype(str)) - set(sc["categories"])
                if bad: warnings.append(f"{c}: unseen categories {sorted(bad)[:3]}")
        if "AMT_INCOME_TOTAL" in df and (pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") <= 0).any(): errors.append("AMT_INCOME_TOTAL must be > 0")
        if "AMT_CREDIT" in df and (pd.to_numeric(df["AMT_CREDIT"], errors="coerce") <= 0).any(): errors.append("AMT_CREDIT must be > 0")
        if not (set(self.cols) & set(df.columns)): errors.append("No recognised model features supplied")
        return errors, warnings

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = add_application_features(df.copy())
        d = d.reindex(columns=self.cols)          # missing features -> NaN (imputed / NaN-native)
        for c, sc in self.schema.items():
            if sc["type"] == "numeric" and c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
        return clean_for_model(d)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        X = self.prepare(df); return np.clip(self.cal.predict(self.pipe.predict_proba(X)[:, 1]), 0, 1)

    @property
    def explainer(self) -> Explainer:
        if self._explainer is None:
            bg = self._background
            if bg is None:
                f = self.paths.processed / "reference_sample.parquet"
                if not f.exists(): raise FileNotFoundError("reference_sample.parquet missing - run training first")
                bg = pd.read_parquet(f)
            self._explainer = Explainer(self.pipe, bg, self.bundle["model_name"])
        return self._explainer

    def assess(self, df: pd.DataFrame, ead: list[float] | None = None, explain: bool = True, top_k: int = 5) -> list[dict]:
        errors, warnings = self.validate(df)
        if errors: raise ValueError("; ".join(errors))
        X = self.prepare(df); p = np.clip(self.cal.predict(self.pipe.predict_proba(X)[:, 1]), 0, 1)
        expl = self.explainer.local(X, top_k) if explain else [None] * len(X)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        out = []
        for i, pdv in enumerate(p):
            band = assign_band(float(pdv), self.bands)
            e = ead[i] if ead is not None and ead[i] is not None else (float(df["AMT_CREDIT"].iloc[i]) if "AMT_CREDIT" in df and pd.notna(df["AMT_CREDIT"].iloc[i]) else None)
            out.append({"applicant_id": None if "SK_ID_CURR" not in df else int(df["SK_ID_CURR"].iloc[i]), "pd": float(pdv),
                        "risk_band": band, "recommendation": self.recs[band], "model_version": self.version, "assessed_at": ts,
                        "ead": e, "ecl_scenarios": ecl_scenarios(float(pdv), e, self.lgd) if e else None,
                        "risk_factors": None if expl[i] is None else expl[i]["top_factors"],
                        "explanation": None if expl[i] is None else expl[i]["text"], "warnings": warnings,
                        "note": "Decision support only; not an automated lending decision. PD is a model estimate, not an observed outcome."})
        return out

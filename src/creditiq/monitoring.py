"""Lightweight reference-vs-current monitoring (data quality, PSI drift, prediction drift)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r = pd.to_numeric(ref, errors="coerce").dropna(); c = pd.to_numeric(cur, errors="coerce").dropna()
    if len(r) < 50 or len(c) < 50 or r.nunique() < 2: return np.nan
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    pr = np.histogram(r, edges)[0] / len(r); pc = np.histogram(c, edges)[0] / len(c)
    pr, pc = np.clip(pr, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(object).fillna("MISSING").value_counts(normalize=True); c = cur.astype(object).fillna("MISSING").value_counts(normalize=True)
    cats = r.index.union(c.index); pr = np.clip(r.reindex(cats).fillna(0).to_numpy(), 1e-4, None); pc = np.clip(c.reindex(cats).fillna(0).to_numpy(), 1e-4, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cfg_mon: dict, exclude=("SK_ID_CURR", "TARGET")) -> pd.DataFrame:
    rows = []
    for c in ref.columns:
        if c in exclude: continue
        if c not in cur.columns:
            rows.append({"feature": c, "type": "schema", "psi": np.nan, "missing_ref": np.nan, "missing_cur": np.nan, "alert": "ALERT", "reason": "column missing in current data"}); continue
        num = pd.api.types.is_numeric_dtype(ref[c])
        psi = psi_numeric(ref[c], cur[c]) if num else psi_categorical(ref[c], cur[c])
        mr, mc = float(ref[c].isna().mean()), float(cur[c].isna().mean())
        level, reason = "OK", []
        if psi == psi and psi >= cfg_mon["psi_alert"]: level = "ALERT"; reason.append(f"PSI {psi:.3f}>={cfg_mon['psi_alert']}")
        elif psi == psi and psi >= cfg_mon["psi_warn"]: level = "WARN"; reason.append(f"PSI {psi:.3f}>={cfg_mon['psi_warn']}")
        if abs(mc - mr) >= cfg_mon["missing_shift_warn"]:
            level = "ALERT" if level == "ALERT" else "WARN"; reason.append(f"missingness shift {mr:.2%}->{mc:.2%}")
        if num and pd.api.types.is_numeric_dtype(cur[c]) is False: level = "ALERT"; reason.append("dtype change")
        rows.append({"feature": c, "type": "numeric" if num else "categorical", "psi": psi, "missing_ref": mr, "missing_cur": mc, "alert": level, "reason": "; ".join(reason)})
    new_cols = [c for c in cur.columns if c not in ref.columns and c not in exclude]
    for c in new_cols: rows.append({"feature": c, "type": "schema", "psi": np.nan, "alert": "WARN", "reason": "new column not in reference"})
    return pd.DataFrame(rows).sort_values(["alert", "psi"], ascending=[True, False]).reset_index(drop=True)


def prediction_monitor(pd_ref: np.ndarray, pd_cur: np.ndarray, bands: dict, cfg_mon: dict) -> dict:
    from .risk_engine import BAND_ORDER, assign_band
    dist = lambda a: pd.Series([assign_band(float(x), bands) for x in a]).value_counts(normalize=True).reindex(BAND_ORDER).fillna(0)
    dr, dc = dist(pd_ref), dist(pd_cur)
    rel = (np.mean(pd_cur) - np.mean(pd_ref)) / max(1e-9, np.mean(pd_ref))
    return {"mean_pd_ref": float(np.mean(pd_ref)), "mean_pd_cur": float(np.mean(pd_cur)), "mean_pd_relative_shift": float(rel),
            "pd_psi": psi_numeric(pd.Series(pd_ref), pd.Series(pd_cur)), "band_share_ref": dr.to_dict(), "band_share_cur": dc.to_dict(),
            "alert": "WARN" if abs(rel) >= cfg_mon["mean_pd_shift_warn"] else "OK",
            "note": "Calibration/performance monitoring requires realised outcomes for the current period; not computed without labels."}

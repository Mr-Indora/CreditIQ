"""Illustrative Expected Credit Loss: ECL = PD x LGD x EAD. Scenario assumptions, NOT measured losses; no IFRS 9 claim."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ecl_scenarios(pd_value: float, ead: float | None, lgd: dict) -> dict | None:
    if ead is None: return None
    return {k: float(pd_value * v * ead) for k, v in lgd.items()}


def portfolio_ecl(scored: pd.DataFrame, lgd: dict, ead_col="AMT_CREDIT", pd_col="PD", band_col="RISK_BAND") -> dict:
    d = scored.dropna(subset=[ead_col]).copy(); res = {"scenario_total": {}, "by_band": {}}
    for k, v in lgd.items():
        d[f"ECL_{k}"] = d[pd_col] * v * d[ead_col]; res["scenario_total"][k] = float(d[f"ECL_{k}"].sum())
    res["total_ead"] = float(d[ead_col].sum())
    res["ecl_rate_of_ead"] = {k: res["scenario_total"][k] / max(1.0, res["total_ead"]) for k in lgd}
    if band_col in d:
        g = d.groupby(band_col).agg(applicants=(pd_col, "size"), ead=(ead_col, "sum"), **{f"ecl_{k}": (f"ECL_{k}", "sum") for k in lgd})
        res["by_band"] = g.reset_index().to_dict("records")
    res["note"] = "Scenario estimates under assumed LGDs; not actual realised losses."
    return res


def sensitivity(scored: pd.DataFrame, lgds=(0.25, 0.35, 0.45, 0.55, 0.65), pd_mult=(0.8, 0.9, 1.0, 1.1, 1.2), ead_col="AMT_CREDIT", pd_col="PD") -> pd.DataFrame:
    d = scored.dropna(subset=[ead_col]); rows = []
    for m in pd_mult:
        for l in lgds:
            rows.append({"pd_multiplier": m, "lgd": l, "portfolio_ecl": float((np.clip(d[pd_col] * m, 0, 1) * l * d[ead_col]).sum())})
    return pd.DataFrame(rows).pivot(index="pd_multiplier", columns="lgd", values="portfolio_ecl")

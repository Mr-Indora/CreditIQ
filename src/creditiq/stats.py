"""Statistical analysis: hypothesis tests with effect sizes, CIs and multiple-testing correction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

Z95 = 1.959964


def wilson_ci(k, n, z=Z95):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj


def bh(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)[::-1]; adj = np.empty(m); prev = 1
    for rank, i in enumerate(order):
        prev = min(prev, p[i] * m / (m - rank)); adj[i] = prev
    return adj


def numeric_test(df, col, target="TARGET", sample=200000, seed=0):
    """Mann-Whitney U (non-normal, heavy tails). Effect size: Cliff's delta (=2*AUC-1) with bootstrap-free
    analytic SE; CI for median difference by bootstrap on a sample."""
    d = df[[col, target]].dropna()
    if len(d) > sample: d = d.sample(sample, random_state=seed)
    a, b = d.loc[d[target] == 1, col].to_numpy(float), d.loc[d[target] == 0, col].to_numpy(float)
    if len(a) < 10 or len(b) < 10: return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b)); delta = 2 * auc - 1
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(200)]
    return {"feature": col, "test": "Mann-Whitney U", "H0": f"{col} distribution is identical for TARGET=1 and TARGET=0",
            "H1": f"{col} distributions differ", "statistic": u, "p_value": p, "effect_size_name": "Cliff's delta",
            "effect_size": delta, "median_target1": float(np.median(a)), "median_target0": float(np.median(b)),
            "median_diff_ci_low": float(np.percentile(diffs, 2.5)), "median_diff_ci_high": float(np.percentile(diffs, 97.5)),
            "n": len(d)}


def categorical_test(df, col, target="TARGET"):
    d = df[[col, target]].copy(); d[col] = d[col].astype(object).where(d[col].notna(), "MISSING")
    ct = pd.crosstab(d[col], d[target])
    if ct.shape[0] < 2 or ct.shape[1] < 2: return None
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum(); v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    rates = ct[1] / ct.sum(axis=1)
    i = int(np.argmax(rates.to_numpy()))            # positional lookup: labels may be bool/mixed (True/False/"MISSING")
    top = rates.index[i]; top_rate = float(rates.to_numpy()[i])
    lo, hi = wilson_ci(int(ct[1].to_numpy()[i]), int(ct.sum(axis=1).to_numpy()[i]))
    return {"feature": col, "test": "Chi-square independence", "H0": f"{col} is independent of TARGET", "H1": f"{col} is associated with TARGET",
            "statistic": chi2, "p_value": p, "effect_size_name": "Cramer's V", "effect_size": v,
            "highest_rate_level": str(top), "highest_rate": top_rate, "highest_rate_ci_low": lo, "highest_rate_ci_high": hi, "n": int(n)}


def run_feature_tests(df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str], target="TARGET") -> pd.DataFrame:
    rows = [r for c in numeric_cols if (r := numeric_test(df, c, target))]
    rows += [r for c in cat_cols if (r := categorical_test(df, c, target))]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_value); out["q_bh"] = bh(out.p_value)
    out["significant_after_correction"] = out.p_holm < 0.05
    def practical(r):
        e = abs(r.effect_size)
        if r.effect_size_name == "Cliff's delta": return "negligible" if e < .147 else "small" if e < .33 else "medium" if e < .474 else "large"
        return "negligible" if e < .1 else "small" if e < .3 else "medium"
    out["practical_significance"] = out.apply(practical, axis=1)
    return out.sort_values("p_value").reset_index(drop=True)


def core_questions(df: pd.DataFrame, tests: pd.DataFrame) -> list[dict]:
    """The four example questions from the brief, answered from the executed tests."""
    qs = {"Is income distributed differently between target groups?": "AMT_INCOME_TOTAL",
          "Is previous credit behaviour associated with repayment difficulties?": "LATE_PAYMENT_RATIO",
          "Does employment duration show an association with the target?": "EMPLOYMENT_DURATION_DAYS",
          "Do financial burden ratios differ?": "ANNUITY_INCOME_RATIO"}
    out = []
    for q, c in qs.items():
        r = tests[tests.feature == c]
        if r.empty: out.append({"question": q, "feature": c, "answer": "feature unavailable"}); continue
        r = r.iloc[0]
        out.append({"question": q, "feature": c, "p_holm": r.p_holm, "effect_size": r.effect_size,
                    "practical_significance": r.practical_significance,
                    "answer": ("Statistically significant" if r.significant_after_correction else "Not significant after correction")
                              + f"; effect is {r.practical_significance}. Association only - not causal."})
    return out

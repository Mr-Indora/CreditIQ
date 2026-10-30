"""EDA figures + auto-generated report (what / why / observation (computed) / interpretation / limitation)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")


def _rate_plot(df, col, bins, labels, fname, fig_dir, title, xlabel):
    d = df[[col, "TARGET"]].dropna(); d["bin"] = pd.cut(d[col], bins=bins, labels=labels)
    g = d.groupby("bin", observed=True).TARGET.agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8)); sns.barplot(data=g, x="bin", y="mean", ax=ax, color="#4c72b0")
    ax.axhline(df.TARGET.mean(), color="r", ls="--", lw=1, label="overall rate"); ax.set(title=title, xlabel=xlabel, ylabel="Historical target rate"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=130); plt.close(fig); return g


def run_eda(df: pd.DataFrame, paths) -> str:
    fd = paths.figures; fd.mkdir(parents=True, exist_ok=True); base = df.TARGET.mean(); sec = []
    def add(title, what, why, obs, interp, lim, fig):
        sec.append(f"### {title}\n![{title}](figures/{fig})\n\n- **What is analysed:** {what}\n- **Why it matters:** {why}\n- **Observation (computed):** {obs}\n- **Business interpretation:** {interp}\n- **Limitations:** {lim}\n")
    # A target distribution
    vc = df.TARGET.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.5)); sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax); ax.set(title="Target distribution", xlabel="TARGET", ylabel="Applicants")
    fig.tight_layout(); fig.savefig(fd / "target_distribution.png", dpi=130); plt.close(fig)
    add("Target distribution", "Share of applicants labelled TARGET=1 (repayment difficulties per the competition's definition).", "Class imbalance drives metric choice, resampling and calibration decisions.",
        f"{int(vc.get(1, 0)):,} of {len(df):,} applicants are TARGET=1 ({base:.2%}).", "Heavily imbalanced -> accuracy is misleading; use ROC-AUC/PR-AUC/KS and calibrated probabilities.",
        "Label reflects the dataset's historical definition, not a universal default event.", "target_distribution.png")
    # B age
    if "AGE_YEARS" in df:
        g = _rate_plot(df, "AGE_YEARS", [0, 25, 35, 45, 55, 120], ["<25", "25-34", "35-44", "45-54", "55+"], "age_vs_target.png", fd, "Target rate by age band", "Age band")
        add("Applicant age", "Historical target rate by age band.", "Age is a common correlate of repayment; also relevant for fairness review.",
            "; ".join(f"{r.bin}: {r['mean']:.2%} (n={r['size']:,})" for _, r in g.iterrows()), "Compare band rates with the overall rate to see where risk concentrates.", "Association only; age bands have unequal sizes.", "age_vs_target.png")
    # C income
    inc = df.AMT_INCOME_TOTAL.clip(upper=df.AMT_INCOME_TOTAL.quantile(.99))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for t, c in [(0, "#4c72b0"), (1, "#c44e52")]: sns.kdeplot(np.log1p(inc[df.TARGET == t]), ax=ax, label=f"TARGET={t}", color=c, common_norm=False)
    ax.set(title="log(income) by target", xlabel="log1p(AMT_INCOME_TOTAL, capped at p99)"); ax.legend(); fig.tight_layout(); fig.savefig(fd / "income_dist.png", dpi=130); plt.close(fig)
    add("Income distribution", "Density of log income per target group (capped at the 99th percentile).", "Income capacity affects affordability.",
        f"Median income TARGET=1: {df.loc[df.TARGET==1,'AMT_INCOME_TOTAL'].median():,.0f}; TARGET=0: {df.loc[df.TARGET==0,'AMT_INCOME_TOTAL'].median():,.0f}.", "Overlapping distributions imply income alone separates groups weakly.", "Income is self-reported/unverified in parts; heavy right tail.", "income_dist.png")
    # D employment
    if "EMPLOYMENT_DURATION_DAYS" in df:
        g = _rate_plot(df, "EMPLOYMENT_DURATION_DAYS", [-1, 365, 1095, 1825, 3650, 100000], ["<1y", "1-3y", "3-5y", "5-10y", "10y+"], "employment_vs_target.png", fd, "Target rate by employment duration", "Employment duration")
        add("Employment duration", "Target rate by length of current employment (365243 sentinel treated as missing).", "Job stability signals income continuity.", "; ".join(f"{r.bin}: {r['mean']:.2%}" for _, r in g.iterrows()),
            "Shorter tenure groups with higher rates would indicate stability risk.", "Sentinel group (unemployed/pensioners) excluded from this chart.", "employment_vs_target.png")
    # E credit & F annuity
    if "CREDIT_INCOME_RATIO" in df:
        q = df.CREDIT_INCOME_RATIO.quantile([.2, .4, .6, .8]).to_numpy(); edges = [-np.inf, *q, np.inf]
        g = _rate_plot(df, "CREDIT_INCOME_RATIO", edges, ["Q1", "Q2", "Q3", "Q4", "Q5"], "credit_income_vs_target.png", fd, "Target rate by credit/income quintile", "Credit/income ratio quintile")
        add("Credit amount relative to income", "Target rate by quintile of AMT_CREDIT/AMT_INCOME_TOTAL.", "Leverage at origination.", "; ".join(f"{r.bin}: {r['mean']:.2%}" for _, r in g.iterrows()), "A rising pattern would support affordability-based rules.", "Ratio ignores other obligations.", "credit_income_vs_target.png")
    if "ANNUITY_INCOME_RATIO" in df:
        q = df.ANNUITY_INCOME_RATIO.quantile([.2, .4, .6, .8]).to_numpy(); g = _rate_plot(df, "ANNUITY_INCOME_RATIO", [-np.inf, *q, np.inf], ["Q1", "Q2", "Q3", "Q4", "Q5"], "annuity_income_vs_target.png", fd, "Target rate by annuity/income quintile", "Annuity/income quintile")
        add("Repayment burden (annuity/income)", "Target rate by instalment burden.", "Payment burden is a classic affordability measure.", "; ".join(f"{r.bin}: {r['mean']:.2%}" for _, r in g.iterrows()), "Higher burden groups warrant closer assessment if rates rise.", "Association, not causation.", "annuity_income_vs_target.png")
    # G household
    if "CNT_FAM_MEMBERS" in df:
        d = df.assign(fam=df.CNT_FAM_MEMBERS.clip(upper=6)).groupby("fam").TARGET.agg(["mean", "size"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.6)); sns.barplot(data=d, x="fam", y="mean", ax=ax, color="#55a868"); ax.set(title="Target rate by family size (capped at 6)", xlabel="Family members", ylabel="Target rate")
        fig.tight_layout(); fig.savefig(fd / "household_vs_target.png", dpi=130); plt.close(fig)
        add("Household characteristics", "Target rate by family size.", "Dependants increase living costs.", "; ".join(f"{int(r.fam)}: {r['mean']:.2%} (n={int(r['size']):,})" for _, r in d.iterrows()), "Check whether large households carry elevated rates.", "Small cells at the tail are noisy.", "household_vs_target.png")
    # H/I/J history
    for col, title, fn in [("BUREAU_PREV_CREDIT_COUNT", "Existing credit exposure (bureau credit count)", "bureau_vs_target.png"), ("LATE_PAYMENT_RATIO", "Repayment behaviour (late payment ratio)", "late_vs_target.png"),
                           ("PREVIOUS_REJECTION_COUNT", "Previous application rejections", "rejections_vs_target.png")]:
        if col in df and df[col].notna().sum() > 100:
            s = df[col]
            if col == "LATE_PAYMENT_RATIO": cats = pd.cut(s, [-0.001, 0, .1, .3, 1.0], labels=["never late", "0-10%", "10-30%", ">30%"])
            else: cats = pd.cut(s, [-1, 0, 1, 3, 1e9], labels=["0", "1", "2-3", "4+"])
            g = df.assign(c=cats).groupby("c", observed=True).TARGET.agg(["mean", "size"]).reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.6)); sns.barplot(data=g, x="c", y="mean", ax=ax, color="#8172b3"); ax.axhline(base, color="r", ls="--", lw=1); ax.set(title=title, xlabel=col, ylabel="Target rate")
            fig.tight_layout(); fig.savefig(fd / fn, dpi=130); plt.close(fig)
            add(title, f"Target rate across levels of {col}.", "Past behaviour is typically the strongest signal in credit data.", "; ".join(f"{r.c}: {r['mean']:.2%} (n={int(r['size']):,})" for _, r in g.iterrows()), "Monotone increases would validate behavioural features.", "Applicants without history are excluded or form separate groups.", fn)
    # K missingness
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6, 5)); sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#ccb974"); ax.set(title="Top-20 missing columns", xlabel="Missing fraction"); fig.tight_layout(); fig.savefig(fd / "missingness.png", dpi=130); plt.close(fig)
    add("Missingness", "Fraction of missing values per column.", "Drives imputation, feature selection and monitoring rules.", f"Highest: {miss.index[0]} at {miss.iloc[0]:.1%}; {int((df.isna().mean() > .5).sum())} columns exceed 50% missing.", "Structural missingness (no bureau record, no card) is informative and handled with HAS_* flags.", "Missing-at-random is not assumed.", "missingness.png")
    # L outliers
    out_rows = []
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1; out_rows.append((c, float(((df[c] > q3 + 3 * iqr)).mean())))
    fig, ax = plt.subplots(figsize=(6, 3.5)); sns.boxplot(data=np.log1p(df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]]), ax=ax); ax.set(title="Amount distributions (log1p)"); fig.tight_layout(); fig.savefig(fd / "outliers.png", dpi=130); plt.close(fig)
    add("Outliers", "Boxplots of main amounts and share above Q3+3*IQR.", "Extreme values can distort linear models and ratios.", "; ".join(f"{c}: {p:.2%} extreme" for c, p in out_rows), "Tree models are robust; linear model uses scaling and median imputation.", "No winsorising applied; extremes may be valid high-value loans.", "outliers.png")
    # M correlations
    num = df.select_dtypes("number").drop(columns=["SK_ID_CURR"], errors="ignore")
    cor = num.corr(numeric_only=True)["TARGET"].drop("TARGET").dropna(); top = cor.reindex(cor.abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(6, 5)); sns.barplot(x=top.values, y=top.index, ax=ax, color="#64b5cd"); ax.set(title="Top correlations with TARGET (Pearson)", xlabel="r"); fig.tight_layout(); fig.savefig(fd / "correlations.png", dpi=130); plt.close(fig)
    add("Correlation with TARGET", "Pearson correlation of numeric features with the label.", "Quick screening of signal strength.", "; ".join(f"{k}: {v:+.3f}" for k, v in top.head(5).items()), "Individually weak correlations are typical; models combine many weak signals.", "Linear correlation misses non-linear effects.", "correlations.png")
    pd.DataFrame({"feature": top.index, "pearson_r": top.values}).to_csv(paths.reports / "top_correlations.csv", index=False)
    # interactive plotly
    if "EXT_SOURCE_2" in df:
        d = df.sample(min(len(df), 20000), random_state=0).assign(TARGET=lambda x: x.TARGET.astype(str))
        px.histogram(d, x="EXT_SOURCE_2", color="TARGET", barmode="overlay", opacity=.6, title="EXT_SOURCE_2 by target").write_html(fd / "ext_source_2.html")
    report = "# EDA Report (auto-generated from executed analysis)\n\n" + f"Rows analysed: {len(df):,}. Overall historical target rate: {base:.2%}.\n\n" + "\n".join(sec)
    (paths.reports / "eda_report.md").write_text(report); return report

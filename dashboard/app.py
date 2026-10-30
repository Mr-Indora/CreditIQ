"""CreditIQ Streamlit dashboard (9 pages). Run: streamlit run dashboard/app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from creditiq.config import get_paths, read_json, load_config

st.set_page_config(page_title="CreditIQ", layout="wide")
paths = get_paths()
PAGES = ["Executive Overview", "Dataset Explorer", "Applicant Risk Assessment", "Portfolio Analytics", "Model Performance",
         "Explainability", "Expected Credit Loss", "Fairness & Governance", "Data & Model Monitoring"]
page = st.sidebar.radio("CreditIQ", PAGES)
st.sidebar.caption("Decision-support tool. PD = model estimate, not an observed outcome. Colab sessions are temporary, not production hosting.")


def J(n):
    f = paths.metrics / n; return read_json(f) if f.exists() else None
def C(p):
    return pd.read_csv(p) if Path(p).exists() else None
def need(x, hint="Run the pipeline stage that produces this artifact first."):
    if x is None: st.info(hint); st.stop()
    return x


@st.cache_resource
def get_engine():
    from creditiq.risk_engine import RiskEngine
    return RiskEngine(paths=paths)

metrics, port, ecl = J("metrics.json"), J("portfolio_summary.json"), J("ecl_summary.json")

if page == "Executive Overview":
    st.title("CreditIQ - Credit Risk Intelligence & Decision Support")
    need(metrics); need(port)
    t = metrics["test_calibrated"]; c = st.columns(5)
    c[0].metric("Holdout ROC-AUC", f"{t['roc_auc']:.3f}"); c[1].metric("PR-AUC", f"{t['pr_auc']:.3f}"); c[2].metric("KS", f"{t['ks']:.3f}")
    c[3].metric("Observed target rate", f"{port['historical_target_rate_observed']:.2%}"); c[4].metric("Mean predicted PD", f"{port['average_predicted_pd_model_estimate']:.2%}")
    st.write(f"Model **{metrics['selected_model']}**, calibration **{metrics['calibration_method']}**, version {metrics['model_version']}.")
    dist = pd.Series(port["risk_segment_distribution"]).rename("share").reset_index().rename(columns={"index": "band"})
    st.plotly_chart(px.bar(dist, x="band", y="share", title="Risk band distribution (holdout)"), use_container_width=True)

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    a = need(C(paths.reports / "dataset_audit.csv")); st.dataframe(a)
    dd = C(paths.reports / "data_dictionary.csv")
    if dd is not None: st.dataframe(dd[dd.table == st.selectbox("Table", sorted(dd.table.unique()))])
    for n in ["relationship_checks.csv", "leakage_audit.csv"]:
        x = C(paths.reports / n)
        if x is not None: st.subheader(n); st.dataframe(x)
    figs = sorted(paths.figures.glob("*_vs_target.png")) + sorted(paths.figures.glob("target_distribution.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]): col.image(str(f))

elif page == "Applicant Risk Assessment":
    st.title("Applicant Risk Assessment")
    eng = get_engine(); hf = need(pd.read_parquet(paths.processed / "holdout_features.parquet") if (paths.processed / "holdout_features.parquet").exists() else None)
    mode = st.radio("Input", ["Sample applicant (holdout)", "Manual entry"], horizontal=True)
    if mode.startswith("Sample"):
        sid = st.selectbox("SK_ID_CURR", hf.SK_ID_CURR.head(500).tolist()); row = hf[hf.SK_ID_CURR == sid].drop(columns=["TARGET"])
    else:
        c = st.columns(3)
        v = {"AMT_INCOME_TOTAL": c[0].number_input("Annual income", 1.0, 1e8, 150000.0), "AMT_CREDIT": c[1].number_input("Loan amount (EAD)", 1.0, 1e8, 500000.0),
             "AMT_ANNUITY": c[2].number_input("Annuity", 1.0, 1e7, 25000.0), "AMT_GOODS_PRICE": c[0].number_input("Goods price", 0.0, 1e8, 450000.0),
             "DAYS_BIRTH": -int(c[1].number_input("Age (years)", 18, 80, 40) * 365.25), "DAYS_EMPLOYED": -int(c[2].number_input("Years employed", 0.0, 50.0, 5.0) * 365.25),
             "CNT_FAM_MEMBERS": c[0].number_input("Family members", 1, 15, 2), "EXT_SOURCE_1": c[1].number_input("EXT_SOURCE_1 (0-1)", 0.0, 1.0, 0.5), "EXT_SOURCE_2": c[2].number_input("EXT_SOURCE_2 (0-1)", 0.0, 1.0, 0.5),
             "EXT_SOURCE_3": c[0].number_input("EXT_SOURCE_3 (0-1)", 0.0, 1.0, 0.5)}
        row = pd.DataFrame([v]); st.caption("Unspecified features are treated as missing and imputed by the pipeline; manual scores are less reliable than full records.")
    if st.button("Assess"):
        try:
            r = eng.assess(row.reset_index(drop=True))[0]
            c = st.columns(3); c[0].metric("Probability of Default", f"{r['pd']:.2%}"); c[1].metric("Risk band", r["risk_band"]); c[2].metric("Recommendation", r["recommendation"])
            st.write(r["explanation"]); f = pd.DataFrame(r["risk_factors"])
            st.plotly_chart(px.bar(f, x="contribution_log_odds", y="feature", orientation="h", title="Top contributions (log-odds, uncalibrated model)"), use_container_width=True)
            if r["ecl_scenarios"]: st.subheader("Illustrative ECL"); st.table(pd.DataFrame([r["ecl_scenarios"]]).round(0))
            for w in r["warnings"]: st.warning(w)
            st.caption(f"Model {r['model_version']} | {r['assessed_at']} | {r['note']}")
        except ValueError as e: st.error(str(e))

elif page == "Portfolio Analytics":
    st.title("Portfolio Analytics (holdout set)"); need(port)
    st.json({k: v for k, v in port.items() if k != "risk_segment_distribution"})
    for n in sorted(paths.reports.glob("portfolio_*.csv")):
        d = pd.read_csv(n); st.subheader(n.stem.replace("portfolio_", "").replace("_", " ")); st.dataframe(d)
        if "observed_target_rate" in d: st.plotly_chart(go.Figure([go.Bar(x=d.iloc[:, 0].astype(str), y=d.observed_target_rate, name="Observed target rate"), go.Bar(x=d.iloc[:, 0].astype(str), y=d.mean_predicted_pd, name="Mean predicted PD")]), use_container_width=True)
    dec = C(paths.reports / "portfolio_risk_deciles.csv")
    if dec is not None: st.plotly_chart(px.line(dec, x="decile", y="cum_capture", markers=True, title="Cumulative target capture"), use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance"); need(metrics)
    for n in ["cv_comparison.csv", "holdout_comparison.csv", "threshold_analysis.csv", "error_analysis.csv", "imbalance_experiment.csv"]:
        x = C(paths.metrics / n)
        if x is not None: st.subheader(n); st.dataframe(x)
    st.subheader("Calibration comparison"); st.dataframe(pd.DataFrame(metrics["calibration_comparison"]))
    cols = st.columns(2)
    for i, n in enumerate(["roc.png", "pr.png", "calibration.png", "confusion.png"]):
        if (paths.figures / n).exists(): cols[i % 2].image(str(paths.figures / n))

elif page == "Explainability":
    st.title("Explainability (SHAP)"); st.caption("Model attributions, not causal effects.")
    imp = need(C(paths.metrics / "shap_global_importance.csv")); st.plotly_chart(px.bar(imp.head(20).iloc[::-1], x="mean_abs_shap", y="feature", orientation="h"), use_container_width=True)
    if (paths.figures / "shap_summary.png").exists(): st.image(str(paths.figures / "shap_summary.png"))

elif page == "Expected Credit Loss":
    st.title("Expected Credit Loss (illustrative)"); st.warning("ECL = PD x LGD x EAD under ASSUMED LGD scenarios. Not realised losses; no IFRS 9 claim.")
    need(ecl); st.json({k: v for k, v in ecl.items() if k != "by_band"}); st.dataframe(pd.DataFrame(ecl["by_band"]))
    s = C(paths.reports / "ecl_sensitivity.csv")
    if s is not None: st.subheader("Sensitivity: portfolio ECL (rows = PD multiplier, cols = LGD)"); st.dataframe(s)

elif page == "Fairness & Governance":
    st.title("Fairness & Governance"); fr = need(C(paths.reports / "fairness_report.csv"))
    st.dataframe(C(paths.reports / "fairness_disparity.csv")); st.dataframe(fr)
    for n in ["MODEL_CARD.md", "FAIRNESS_REPORT.md", "LIMITATIONS.md"]:
        f = paths.docs / n
        if f.exists():
            with st.expander(n): st.markdown(f.read_text())

elif page == "Data & Model Monitoring":
    st.title("Data & Model Monitoring"); st.caption("Reference = training sample; current = holdout (random split, so little drift is expected). Use new-period data for real monitoring.")
    d = need(C(paths.metrics / "drift_report.csv")); st.dataframe(d); pm = J("prediction_monitoring.json")
    if pm: st.json(pm)

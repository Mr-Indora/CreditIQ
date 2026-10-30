import numpy as np, pandas as pd, pytest
from creditiq.ecl import ecl_scenarios, portfolio_ecl
from creditiq.monitoring import drift_report, psi_numeric
from creditiq.risk_engine import RiskEngine, assign_band
from creditiq import stats

BANDS = {"LOW": [0, .05], "MEDIUM": [.05, .15], "HIGH": [.15, .30], "VERY_HIGH": [.30, 1.01]}


def test_band_assignment_edges():
    assert [assign_band(x, BANDS) for x in (0.0, 0.0499, 0.05, 0.149, 0.15, 0.2999, 0.30, 1.0)] == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "VERY_HIGH", "VERY_HIGH"]


def test_ecl_math():
    r = ecl_scenarios(0.10, 100000, {"optimistic": .25, "base": .45, "conservative": .65})
    assert abs(r["base"] - 4500) < 1e-6 and r["optimistic"] < r["base"] < r["conservative"]
    s = pd.DataFrame({"PD": [.1, .2], "AMT_CREDIT": [1000., 2000.], "RISK_BAND": ["MEDIUM", "HIGH"]})
    assert abs(portfolio_ecl(s, {"base": .5})["scenario_total"]["base"] - 250) < 1e-9


def test_engine_assess_end_to_end(project):
    e = RiskEngine(paths=project)
    ho = pd.read_parquet(project.processed / "holdout_features.parquet").head(3)
    out = e.assess(ho)
    assert len(out) == 3
    for r in out:
        assert 0 <= r["pd"] <= 1 and r["risk_band"] in BANDS and r["model_version"] == e.version and r["assessed_at"]
        assert r["recommendation"] in e.recs.values() and len(r["risk_factors"]) == 5 and set(r["ecl_scenarios"]) == {"optimistic", "base", "conservative"}


def test_engine_validation_errors(project):
    e = RiskEngine(paths=project)
    with pytest.raises(ValueError): e.assess(pd.DataFrame([{"AMT_INCOME_TOTAL": -5, "AMT_CREDIT": 100}]))
    with pytest.raises(ValueError): e.assess(pd.DataFrame([{"NOT_A_FEATURE": 1}]))
    with pytest.raises(ValueError): e.set_bands({"LOW": [0, .1], "MEDIUM": [.05, .15], "HIGH": [.15, .3], "VERY_HIGH": [.3, 1.01]})


def test_engine_partial_input_works(project):
    e = RiskEngine(paths=project)
    r = e.assess(pd.DataFrame([{"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 20000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2000, "EXT_SOURCE_2": .5}]), explain=False)[0]
    assert 0 <= r["pd"] <= 1


def test_psi_detects_shift_and_stable():
    rng = np.random.default_rng(0); a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_drift_report_flags_simulated_shift(project):
    ref = pd.read_parquet(project.processed / "reference_sample.parquet"); cur = ref.copy()
    cur["AMT_CREDIT"] = cur["AMT_CREDIT"] * 3; cur["EXT_SOURCE_2"] = np.nan
    rep = drift_report(ref, cur, {"psi_warn": .1, "psi_alert": .25, "missing_shift_warn": .05, "mean_pd_shift_warn": .2}).set_index("feature")
    assert rep.loc["AMT_CREDIT", "alert"] == "ALERT" and rep.loc["EXT_SOURCE_2", "alert"] in ("WARN", "ALERT")


def test_stats_corrections_and_tests():
    p = np.array([.001, .01, .04, .5]); assert (stats.holm(p) >= p).all() and stats.holm(p).max() <= 1
    rng = np.random.default_rng(0); df = pd.DataFrame({"TARGET": rng.integers(0, 2, 3000)}); df["x"] = rng.normal(size=3000) + df.TARGET * 1.0
    r = stats.numeric_test(df, "x"); assert r["p_value"] < 1e-10 and r["effect_size"] != 0
    lo, hi = stats.wilson_ci(50, 1000); assert lo < .05 < hi


def test_reports_generated_and_flag_synthetic(project):
    txt = (project.docs / "MODEL_CARD.md").read_text()
    assert "SYNTHETIC" in txt                                  # honest labelling of smoke-test run
    assert "Not generated" in (project.docs / "RESUME_BULLETS.md").read_text()


def test_api_endpoints(project):
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/model/info").json()["model_version"]
    body = {"applicant_id": 1, "features": {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 20000, "DAYS_BIRTH": -14000, "EXT_SOURCE_2": .4}, "ead": 450000}
    r = c.post("/predict", json=body); assert r.status_code == 200 and 0 <= r.json()["pd"] <= 1 and r.json()["risk_band"]
    assert c.post("/predict", json={"features": {"AMT_INCOME_TOTAL": -1, "AMT_CREDIT": 5}}).status_code == 422
    rb = c.post("/predict/batch", json={"applicants": [body, body]}); assert rb.status_code == 200 and rb.json()["count"] == 2
    for ep in ["/risk/portfolio", "/model/metrics", "/monitoring/drift"]: assert c.get(ep).status_code == 200, ep


def test_dashboard_pages_render(project):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception
    for pg in ["Model Performance", "Portfolio Analytics", "Expected Credit Loss", "Fairness & Governance", "Data & Model Monitoring", "Explainability", "Dataset Explorer"]:
        at.sidebar.radio[0].set_value(pg).run(); assert not at.exception, (pg, at.exception)

import numpy as np, pandas as pd
from creditiq import artifacts as art
from creditiq.calibration import Calibrator, select_calibrator
from creditiq.evaluation import all_metrics, decile_table, ks_stat
from creditiq.modeling import build_pipeline, prepare_xy


def test_artifacts_exist(project):
    for f in [project.models / art.BUNDLE_NAME, project.metrics / "metrics.json", project.metrics / "cv_comparison.csv", project.metrics / "holdout_comparison.csv",
              project.figures / "roc.png", project.figures / "calibration.png", project.figures / "shap_summary.png"]:
        assert f.exists(), f


def test_bundle_contents_and_probabilities(project):
    b = art.load_bundle(project.models / art.BUNDLE_NAME, verify=False)
    ho = pd.read_parquet(project.processed / "holdout_features.parquet")
    from creditiq.risk_engine import RiskEngine
    e = RiskEngine(paths=project)
    p = e.predict_pd(ho)
    assert p.shape == (len(ho),) and np.all((p >= 0) & (p <= 1)) and np.isfinite(p).all()
    assert b["schema"] and b["raw_columns"] and "TARGET" not in b["raw_columns"] and "SK_ID_CURR" not in b["raw_columns"]
    assert "CODE_GENDER" not in b["raw_columns"]           # sensitive attribute excluded from model inputs


def test_holdout_untouched_by_training(project):
    ho = pd.read_parquet(project.processed / "holdout_features.parquet"); tr_ids = pd.read_parquet(project.processed / "train_ids.parquet")
    assert not (set(ho.SK_ID_CURR) & set(tr_ids.SK_ID_CURR))


def test_reproducibility(project, cfg):
    df = pd.read_parquet(project.processed / "features_train.parquet").head(1500)
    X, y, _ = prepare_xy(df, cfg)
    p1 = build_pipeline("lightgbm", cfg).fit(X, y).predict_proba(X)[:, 1]
    p2 = build_pipeline("lightgbm", cfg).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p1, p2)


def test_manifest_checksum_detects_tamper(project):
    import pytest
    art.write_manifest(project, "1.0.0")
    f = project.models / art.BUNDLE_NAME; data = f.read_bytes()
    try:
        f.write_bytes(data + b"x")
        with pytest.raises(ValueError): art.load_bundle(f)
    finally:
        f.write_bytes(data)


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .35, .8])
    m = all_metrics(y, p, 0.5)
    assert abs(m["roc_auc"] - 0.75) < 1e-9 and m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 1 and m["tn"] == 2
    assert abs(m["specificity"] - 1.0) < 1e-9 and abs(ks_stat(y, p) - 0.5) < 1e-9


def test_calibrators_improve_inflated_probabilities():
    rng = np.random.default_rng(0); true = rng.uniform(0, .3, 4000); y = (rng.random(4000) < true).astype(int); inflated = np.clip(true * 3, 0, 1)
    cal, rows = select_calibrator(inflated, y)
    assert cal.method in ("platt", "isotonic")
    assert abs(cal.predict(inflated).mean() - y.mean()) < abs(inflated.mean() - y.mean())


def test_decile_table_monotone_capture():
    rng = np.random.default_rng(1); p = rng.uniform(size=2000); y = (rng.random(2000) < p * .3).astype(int)
    t = decile_table(y, p)
    assert t.cum_capture.is_monotonic_increasing and abs(t.cum_capture.iloc[-1] - 1) < 1e-9 and t.observed_rate.iloc[0] > t.observed_rate.iloc[-1]


def test_no_resampling_of_eval_data_smote_flag_off(cfg):
    assert cfg["use_smote"] is False

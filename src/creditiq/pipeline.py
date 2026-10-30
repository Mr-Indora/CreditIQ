"""Stage orchestrator with checkpointing. Usage: python -m creditiq.pipeline --stage all [--synthetic]
Stages: ingest, audit, aggregate, features, eda, stats, train, analytics, report, export."""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
from . import audit, data_io, eda, stats
from .aggregate import build_feature_table, run_aggregations, run_analysis_queries
from .config import get_paths, is_done, load_config, mark_done, write_json
from .features import FEATURE_DOCS, model_feature_columns

STAGES = ["ingest", "audit", "aggregate", "features", "eda", "stats", "train", "analytics", "report", "export"]


def stage_ingest(paths, cfg, synthetic=False, n_train=4000, force=False):
    if synthetic:
        from .synthetic import make_synthetic
        make_synthetic(paths.raw, n_train=n_train, n_test=max(200, n_train // 8), seed=cfg["seed"])
        print("WARNING: synthetic data generated - for smoke tests only; never report its metrics.")
    raw = data_io.find_raw_dir(paths) or data_io.download_dataset(paths)
    if raw is None: raise SystemExit("Dataset unavailable - see instructions above.")
    v = data_io.validate_files(raw); v.to_csv(paths.reports / "file_validation.csv", index=False)
    data_io.csv_to_parquet(paths, raw, force=force)
    return {"raw_dir": str(raw), "synthetic": synthetic}


def stage_audit(paths, cfg):
    a = audit.dataset_audit(paths); audit.relationship_checks(paths); audit.build_data_dictionary(paths); audit.leakage_audit(paths)
    run_analysis_queries(paths); return {"tables": len(a)}


def stage_aggregate(paths, cfg, force=False): run_aggregations(paths, force=force); return {}


def stage_features(paths, cfg):
    tr = build_feature_table(paths, "train"); tr.to_parquet(paths.processed / "features_train.parquet", index=False)
    te = build_feature_table(paths, "test"); te.to_parquet(paths.processed / "features_test.parquet", index=False)
    cols = model_feature_columns(tr, cfg)
    audit.feature_availability_report(cols, cfg).to_csv(paths.reports / "feature_availability.csv", index=False)
    pd.DataFrame([{"feature": k, "formula": v[0], "inputs": v[1], "business_meaning": v[2], "missing_handling": v[3]} for k, v in FEATURE_DOCS.items()]).to_csv(paths.reports / "feature_documentation.csv", index=False)
    # train/test consistency
    cons = {"train_cols_missing_in_test": sorted(set(tr.columns) - set(te.columns) - {"TARGET"}), "test_cols_missing_in_train": sorted(set(te.columns) - set(tr.columns)),
            "train_rows": len(tr), "test_rows": len(te), "unique_ids_train": bool(tr.SK_ID_CURR.is_unique), "target_values": sorted(tr.TARGET.unique().tolist())}
    assert not cons["train_cols_missing_in_test"] and not cons["test_cols_missing_in_train"], cons
    write_json(cons, paths.metrics / "train_test_consistency.json")
    return {"rows": len(tr), "features": len(cols)}


def _load_train(paths): return pd.read_parquet(paths.processed / "features_train.parquet")


def stage_eda(paths, cfg): eda.run_eda(_load_train(paths), paths); return {}


def stage_stats(paths, cfg):
    df = _load_train(paths)
    num = [c for c in df.select_dtypes("number").columns if c not in ("SK_ID_CURR", "TARGET") and df[c].nunique() > 5 and df[c].notna().mean() > .3][:80]
    cat = [c for c in df.columns if c not in num and c not in ("SK_ID_CURR", "TARGET") and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() < 60]
    cat += [c for c in df.select_dtypes("number").columns if c.startswith(("FLAG_", "HAS_")) or c in ("REGION_RATING_CLIENT",)]
    t = stats.run_feature_tests(df, num, cat); t.to_csv(paths.reports / "statistical_tests.csv", index=False)
    write_json(stats.core_questions(df, t), paths.reports / "core_questions.json"); return {"tests": len(t)}


def stage_train(paths, cfg, force_cv=False):
    from .modeling import train_and_evaluate
    m = train_and_evaluate(_load_train(paths), cfg, paths, force_cv=force_cv); return {"model": m["selected_model"], "test_auc": m["test_calibrated"]["roc_auc"]}


def stage_analytics(paths, cfg):
    from . import ecl, fairness, monitoring, portfolio
    from .risk_engine import RiskEngine
    from .config import read_json
    eng = RiskEngine(paths=paths); metrics = read_json(paths.metrics / "metrics.json")
    hf = pd.read_parquet(paths.processed / "holdout_features.parquet"); scored = pd.read_parquet(paths.processed / "holdout_scored.parquet")
    # explainability (global) on a holdout sample
    X = eng.prepare(hf.sample(min(len(hf), 2000), random_state=cfg["seed"]))
    imp = eng.explainer.global_importance(X); imp.to_csv(paths.metrics / "shap_global_importance.csv", index=False)
    eng.explainer.summary_plot(X, paths.figures / "shap_summary.png")
    # portfolio
    scored = portfolio.add_bands(scored, eng.bands); scored.to_parquet(paths.processed / "holdout_scored.parquet", index=False)
    write_json(portfolio.summary(scored, eng.bands, metrics), paths.metrics / "portfolio_summary.json")
    for k, v in portfolio.build_segment_reports(scored, eng.bands).items(): v.to_csv(paths.reports / f"portfolio_{k}.csv", index=False)
    # ECL
    write_json(ecl.portfolio_ecl(scored, eng.lgd), paths.metrics / "ecl_summary.json"); ecl.sensitivity(scored).to_csv(paths.reports / "ecl_sensitivity.csv")
    # fairness
    rep = fairness.fairness_report(scored, metrics["decision_threshold"], cfg["fairness"]["min_group_size"])
    rep.to_csv(paths.reports / "fairness_report.csv", index=False); fairness.disparity_summary(rep).to_csv(paths.reports / "fairness_disparity.csv", index=False)
    # monitoring: reference = fit-split sample, current = holdout (random split -> expected stable; real use: new period data)
    ref = pd.read_parquet(paths.processed / "reference_sample.parquet"); cur = eng.prepare(hf)
    dr = monitoring.drift_report(ref, cur, cfg["monitoring"]); dr.to_csv(paths.metrics / "drift_report.csv", index=False)
    pm = monitoring.prediction_monitor(eng.predict_pd(ref), scored.PD.to_numpy(), eng.bands, cfg["monitoring"]); write_json(pm, paths.metrics / "prediction_monitoring.json")
    return {"drift_alerts": int((dr.alert == "ALERT").sum())}


def stage_report(paths, cfg):
    from .reporting import generate_reports
    generate_reports(paths, cfg); return {}


def stage_export(paths, cfg):
    from .artifacts import export_zip
    z = export_zip(paths, "1.0.0"); print("Export bundle:", z); return {"zip": str(z)}


def run(stage="all", root=None, synthetic=False, n_train=4000, force=False):
    paths = get_paths(root); cfg = load_config(paths)
    fns = {"ingest": lambda: stage_ingest(paths, cfg, synthetic, n_train, force), "audit": lambda: stage_audit(paths, cfg),
           "aggregate": lambda: stage_aggregate(paths, cfg, force), "features": lambda: stage_features(paths, cfg),
           "eda": lambda: stage_eda(paths, cfg), "stats": lambda: stage_stats(paths, cfg), "train": lambda: stage_train(paths, cfg, force),
           "analytics": lambda: stage_analytics(paths, cfg), "report": lambda: stage_report(paths, cfg), "export": lambda: stage_export(paths, cfg)}
    todo = STAGES if stage == "all" else [stage]
    for s in todo:
        if stage == "all" and is_done(paths, s) and not force and s != "export":
            print(f"[skip] {s} already completed"); continue
        t0 = time.time(); print(f"[run ] {s} ...")
        info = fns[s](); mark_done(paths, s, {"seconds": round(time.time() - t0, 1), **(info or {})}); print(f"[done] {s} in {time.time()-t0:.0f}s {info or ''}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all", *STAGES]); ap.add_argument("--root"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-train", type=int, default=4000); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    run(a.stage, a.root, a.synthetic, a.n_train, a.force)

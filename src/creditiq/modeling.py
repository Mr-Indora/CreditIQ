"""Model zoo, CV comparison, final fit, calibration, holdout evaluation."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import artifacts as art
from .calibration import select_calibrator
from .config import write_json
from .evaluation import (all_metrics, best_f1_threshold, decile_table, error_analysis, plot_calibration, plot_confusion,
                         plot_roc_pr, threshold_table)
from .features import add_application_features, clean_for_model, make_preprocessor, model_feature_columns

KIND = {"dummy": "tree_imputed", "logreg": "linear", "random_forest": "tree_imputed", "xgboost": "tree_nan", "lightgbm": "tree_nan"}
INTERPRETABILITY = {"dummy": "n/a", "logreg": "high (coefficients)", "random_forest": "medium", "xgboost": "medium (SHAP)", "lightgbm": "medium (SHAP)"}


def make_estimator(name: str, cfg: dict, pos_weight: float = 1.0, params: dict | None = None):
    seed = cfg["seed"]; params = params or {}
    if name == "dummy": return DummyClassifier(strategy="prior")
    if name == "logreg": return LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
                             reg_lambda=5, tree_method="hist", scale_pos_weight=pos_weight, eval_metric="auc", random_state=seed, n_jobs=-1)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        p = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1, verbose=-1)
        p.update(params); return LGBMClassifier(**p)
    raise ValueError(name)


def build_pipeline(name: str, cfg: dict, pos_weight=1.0, params=None) -> Pipeline:
    prep = make_preprocessor(KIND[name], cfg); est = make_estimator(name, cfg, pos_weight, params)
    if cfg.get("use_smote") and KIND[name] != "tree_nan" and name != "dummy":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline   # SMOTE runs only on training folds inside the pipeline
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=cfg["seed"])), ("clf", est)])
    return Pipeline([("prep", prep), ("clf", est)])


def prepare_xy(df: pd.DataFrame, cfg: dict):
    cols = model_feature_columns(df, cfg)
    return clean_for_model(df[cols]), df["TARGET"].astype(int).to_numpy(), cols


def cv_compare(X, y, cfg, models=None) -> pd.DataFrame:
    """Stratified CV on the development set (optionally a stratified subsample for Colab runtime)."""
    models = models or cfg["models"]; n = cfg["cv"].get("sample_n")
    if n and len(X) > n:
        X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=cfg["seed"])
    skf = StratifiedKFold(cfg["cv"]["folds"], shuffle=True, random_state=cfg["seed"])
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score", "log_loss": "neg_log_loss"}
    rows = []
    for m in models:
        t0 = time.time()
        try:
            r = cross_validate(build_pipeline(m, cfg), X, y, cv=skf, scoring=scoring, n_jobs=1, error_score="raise")
        except ImportError as e:
            print(f"skip {m}: {e}"); continue
        rows.append({"model": m, **{f"cv_{k}_mean": (-1 if k in ("brier", "log_loss") else 1) * r[f"test_{k}"].mean() for k in scoring},
                     "cv_roc_auc_std": r["test_roc_auc"].std(), "fit_seconds_per_fold": float(np.mean(r["fit_time"])),
                     "interpretability": INTERPRETABILITY[m], "cv_rows": len(X)})
        print(f"{m}: AUC {rows[-1]['cv_roc_auc_mean']:.4f}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def imbalance_experiment(X, y, cfg, n=60000) -> pd.DataFrame:
    """LightGBM with/without positive-class weighting (class-weighting study). Resampling is off by default."""
    try: import lightgbm  # noqa
    except ImportError: return pd.DataFrame()
    if len(X) > n: X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=cfg["seed"])
    skf = StratifiedKFold(3, shuffle=True, random_state=cfg["seed"]); spw = (y == 0).sum() / max(1, (y == 1).sum()); rows = []
    for label, w in [("no_weighting", 1.0), ("balanced_weight", spw)]:
        r = cross_validate(build_pipeline("lightgbm", cfg, pos_weight=w), X, y, cv=skf,
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"})
        rows.append({"setup": label, "roc_auc": r["test_roc_auc"].mean(), "pr_auc": r["test_pr_auc"].mean(), "brier": -r["test_brier"].mean()})
    return pd.DataFrame(rows)


def tune_lightgbm(X, y, cfg, trials: int):
    import optuna
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if len(X) > 80000: X, _, y, _ = train_test_split(X, y, train_size=80000, stratify=y, random_state=cfg["seed"])
    def obj(t):
        p = {"num_leaves": t.suggest_int("num_leaves", 15, 80), "min_child_samples": t.suggest_int("min_child_samples", 30, 300),
             "learning_rate": t.suggest_float("learning_rate", 0.01, 0.08, log=True), "colsample_bytree": t.suggest_float("colsample_bytree", .4, .9),
             "reg_lambda": t.suggest_float("reg_lambda", 1, 30, log=True),
             "n_estimators": t.suggest_int("n_estimators", 300, 1200, step=100)}
        return cross_val_score(build_pipeline("lightgbm", cfg, params=p), X, y, cv=3, scoring="roc_auc").mean()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"])); st.optimize(obj, n_trials=trials)
    return st.best_params


def train_and_evaluate(df: pd.DataFrame, cfg: dict, paths, version: str = "1.0.0", force_cv: bool = False, audit_cols=("CODE_GENDER",)) -> dict:
    """Full modelling run. Splits: untouched test | development = (fit | calibration)."""
    seed = cfg["seed"]
    X_all, y_all, cols = prepare_xy(df, cfg)
    ids = df["SK_ID_CURR"].to_numpy()
    idx = np.arange(len(df))
    dev_i, test_i = train_test_split(idx, test_size=cfg["split"]["test_size"], stratify=y_all, random_state=seed)
    fit_i, cal_i = train_test_split(dev_i, test_size=cfg["split"]["calibration_size"], stratify=y_all[dev_i], random_state=seed)
    X_dev, y_dev = X_all.iloc[dev_i], y_all[dev_i]

    # 1) CV comparison on development set (test set never touched)
    cv_file = paths.metrics / "cv_comparison.csv"
    if cv_file.exists() and not force_cv:
        cv = pd.read_csv(cv_file)
    else:
        cv = cv_compare(X_dev, y_dev, cfg); cv.to_csv(cv_file, index=False)
        imb = imbalance_experiment(X_dev, y_dev, cfg)
        if len(imb): imb.to_csv(paths.metrics / "imbalance_experiment.csv", index=False)
    sel = cfg["selected_model"]
    if sel == "auto": sel = cv[cv.model != "dummy"].iloc[0].model
    params = None
    if sel == "lightgbm" and cfg["tuning"]["optuna_trials"] > 0:
        params = tune_lightgbm(X_dev, y_dev, cfg, cfg["tuning"]["optuna_trials"])
    # 2) fit on fit-split
    pos_w = 1.0
    pipe = build_pipeline(sel, cfg, pos_weight=pos_w, params=params).fit(X_all.iloc[fit_i], y_all[fit_i])
    p_cal_raw = pipe.predict_proba(X_all.iloc[cal_i])[:, 1]; p_test_raw = pipe.predict_proba(X_all.iloc[test_i])[:, 1]
    # 3) calibration (fit on calibration split only)
    calibrator, cal_rows = select_calibrator(p_cal_raw, y_all[cal_i], seed)
    p_cal = calibrator.predict(p_cal_raw); p_test = calibrator.predict(p_test_raw)
    thr = best_f1_threshold(y_all[cal_i], p_cal)   # selected on calibration split, reported on test
    y_test = y_all[test_i]
    metrics = {"selected_model": sel, "model_version": version, "decision_threshold": thr, "calibration_method": calibrator.method,
               "calibration_comparison": cal_rows,
               "test_raw": all_metrics(y_test, p_test_raw, best_f1_threshold(y_all[cal_i], p_cal_raw)),
               "test_calibrated": all_metrics(y_test, p_test, thr),
               "split_sizes": {"fit": len(fit_i), "calibration": len(cal_i), "test": len(test_i), "features_in": len(cols)},
               "tuned_params": params, "seed": seed}
    # baselines on the same holdout (for the comparison table)
    base = {}
    for m in [x for x in cfg["models"] if x != sel]:
        try:
            pb = build_pipeline(m, cfg).fit(X_all.iloc[fit_i], y_all[fit_i]); pt = pb.predict_proba(X_all.iloc[test_i])[:, 1]
            pc = pb.predict_proba(X_all.iloc[cal_i])[:, 1]
            base[m] = (pt, best_f1_threshold(y_all[cal_i], pc) if m != "dummy" else 0.5)
        except ImportError: continue
    holdout_rows = [{"model": m, **{k: v for k, v in all_metrics(y_test, pt, t).items() if k in ("roc_auc", "pr_auc", "ks", "brier", "log_loss", "recall", "precision", "f1")}} for m, (pt, t) in base.items()]
    holdout_rows.append({"model": f"{sel} (selected, raw)", **{k: v for k, v in metrics["test_raw"].items() if k in ("roc_auc", "pr_auc", "ks", "brier", "log_loss", "recall", "precision", "f1")}})
    holdout_rows.append({"model": f"{sel} (selected, calibrated={calibrator.method})", **{k: v for k, v in metrics["test_calibrated"].items() if k in ("roc_auc", "pr_auc", "ks", "brier", "log_loss", "recall", "precision", "f1")}})
    pd.DataFrame(holdout_rows).to_csv(paths.metrics / "holdout_comparison.csv", index=False)
    # tables & plots
    threshold_table(y_test, p_test).to_csv(paths.metrics / "threshold_analysis.csv", index=False)
    decile_table(y_test, p_test).to_csv(paths.metrics / "risk_deciles_test.csv", index=False)
    error_analysis(X_all.iloc[test_i], y_test, p_test, thr).to_csv(paths.metrics / "error_analysis.csv", index=False)
    fig = paths.figures
    curves = {f"{sel} raw": (y_test, p_test_raw), f"{sel} calibrated": (y_test, p_test), **{m: (y_test, pt) for m, (pt, _) in base.items() if m != "dummy"}}
    plot_roc_pr(curves, fig / "roc.png", fig / "pr.png")
    plot_calibration({f"{sel} raw": (y_test, p_test_raw), f"{sel} {calibrator.method}": (y_test, p_test)}, fig / "calibration.png")
    plot_confusion(metrics["test_calibrated"], fig / "confusion.png")
    write_json(metrics, paths.metrics / "metrics.json")
    # holdout scoring frame for portfolio / fairness / monitoring / dashboard
    keep = ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "EMPLOYMENT_DURATION_DAYS", "AGE_YEARS", "LATE_PAYMENT_RATIO", "PREVIOUS_REJECTION_COUNT", "BUREAU_PREV_CREDIT_COUNT", *audit_cols]
    scored = df.iloc[test_i][[c for c in dict.fromkeys(keep) if c in df.columns]].copy()
    scored["PD_RAW"] = p_test_raw; scored["PD"] = p_test; scored.to_parquet(paths.processed / "holdout_scored.parquet", index=False)
    df.iloc[test_i].to_parquet(paths.processed / "holdout_features.parquet", index=False)
    pd.DataFrame({"SK_ID_CURR": ids[fit_i]}).to_parquet(paths.processed / "train_ids.parquet", index=False)
    X_all.iloc[fit_i].sample(min(len(fit_i), 20000), random_state=seed).to_parquet(paths.processed / "reference_sample.parquet", index=False)
    # bundle
    bundle = {"model_version": version, "model_name": sel, "pipeline": pipe, "calibrator": calibrator, "raw_columns": cols,
              "schema": art.build_schema(X_all.iloc[fit_i]), "decision_threshold": thr, "risk_bands": cfg["risk_bands"],
              "recommendations": cfg["recommendations"], "lgd_scenarios": cfg["ecl"]["lgd_scenarios"], "metrics": metrics,
              "dropped_by_selector": list(pipe.named_steps["prep"].named_steps["select"].dropped_), "seed": seed}
    art.save_bundle(paths, bundle)
    return metrics

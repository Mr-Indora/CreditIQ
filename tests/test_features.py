import numpy as np, pandas as pd
from creditiq.features import FeatureSelector, add_application_features, safe_div, make_preprocessor


def test_safe_div_zero_and_inf():
    out = safe_div(pd.Series([1, 2, 3]), pd.Series([0, 2, -1]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1 and np.isnan(out.iloc[2])


def test_ratios_and_sentinel():
    df = pd.DataFrame({"AMT_CREDIT": [100.0, 50.0], "AMT_INCOME_TOTAL": [50.0, 0.0], "AMT_ANNUITY": [10.0, 5.0], "CNT_FAM_MEMBERS": [2, 1],
                       "DAYS_EMPLOYED": [-365, 365243], "DAYS_BIRTH": [-7305, -10000]})
    o = add_application_features(df)
    assert o.CREDIT_INCOME_RATIO.iloc[0] == 2.0 and np.isnan(o.CREDIT_INCOME_RATIO.iloc[1])
    assert o.EMPLOYMENT_DURATION_DAYS.iloc[0] == 365 and np.isnan(o.EMPLOYMENT_DURATION_DAYS.iloc[1])
    assert list(o.DAYS_EMPLOYED_ANOM) == [0, 1]
    assert abs(o.AGE_YEARS.iloc[0] - 20.0) < 0.01
    assert not np.isinf(o.select_dtypes("number").to_numpy(dtype=float)).any()


def test_add_features_idempotent():
    df = pd.DataFrame({"AMT_CREDIT": [100.0], "AMT_INCOME_TOTAL": [50.0], "DAYS_EMPLOYED": [-10]})
    once = add_application_features(df); twice = add_application_features(once)
    assert once.CREDIT_INCOME_RATIO.iloc[0] == twice.CREDIT_INCOME_RATIO.iloc[0]


def test_selector_drops_sparse_constant_and_correlated():
    rng = np.random.default_rng(0); a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "a_copy": a * 2 + 1e-6, "const": 1.0, "sparse": np.where(rng.random(500) < .9, np.nan, 1.0), "b": rng.normal(size=500)})
    s = FeatureSelector(0.75, 0.98).fit(X)
    assert set(s.keep_) == {"a", "b"}


def test_preprocessor_fit_on_train_only_handles_unseen_category():
    tr = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0] * 10, "c": ["u", "v", "u", np.nan] * 10})
    te = pd.DataFrame({"x": [np.nan], "c": ["NEW"]})
    for kind in ["linear", "tree_nan", "tree_imputed"]:
        p = make_preprocessor(kind, {"seed": 1, "features": {"max_missing_frac": .9, "corr_prune_threshold": .99}}).fit(tr)
        out = p.transform(te)
        assert len(out) == 1
        if kind != "tree_nan": assert not out.isna().any().any()

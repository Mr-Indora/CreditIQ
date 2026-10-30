"""Feature engineering, leakage rules and sklearn preprocessing (fit on training data only)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ID_COL, TARGET = "SK_ID_CURR", "TARGET"
DAYS_EMPLOYED_SENTINEL = 365243

FEATURE_DOCS = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT / AMT_INCOME_TOTAL", "AMT_CREDIT, AMT_INCOME_TOTAL", "Loan size relative to annual income", "NaN if income<=0"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY / AMT_INCOME_TOTAL", "AMT_ANNUITY, AMT_INCOME_TOTAL", "Repayment burden vs income", "NaN if income<=0"),
    "CREDIT_ANNUITY_RATIO": ("AMT_CREDIT / AMT_ANNUITY", "AMT_CREDIT, AMT_ANNUITY", "Implied loan term in payment periods", "NaN if annuity<=0"),
    "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL, CNT_FAM_MEMBERS", "Household income per member", "NaN if members<=0"),
    "EMPLOYMENT_DURATION_DAYS": ("-DAYS_EMPLOYED (365243 sentinel -> NaN)", "DAYS_EMPLOYED", "Length of current employment", "Sentinel -> NaN + DAYS_EMPLOYED_ANOM flag"),
    "EMPLOYMENT_AGE_RATIO": ("DAYS_EMPLOYED / DAYS_BIRTH", "DAYS_EMPLOYED, DAYS_BIRTH", "Share of life employed", "NaN where employment unknown"),
    "AGE_YEARS": ("-DAYS_BIRTH / 365.25", "DAYS_BIRTH", "Applicant age at application", "n/a"),
    "ACTIVE_CREDIT_RATIO": ("active bureau credits / all bureau credits", "bureau.CREDIT_ACTIVE", "Share of currently open external credits", "NaN if no bureau record"),
    "PREVIOUS_APPROVAL_RATIO": ("approved prev. apps / all prev. apps", "previous_application.NAME_CONTRACT_STATUS", "Historical approval propensity", "NaN if no history"),
    "AVERAGE_CREDIT_DEBT": ("mean(AMT_CREDIT_SUM_DEBT)", "bureau", "Average outstanding external debt", "NaN if no bureau"),
    "OVERDUE_ACCOUNT_RATIO": ("share of bureau credits with overdue days/amount", "bureau", "Overdue incidence", "NaN if no bureau"),
    "LATE_PAYMENT_RATIO": ("share of instalments paid after due date", "installments_payments", "Payment punctuality", "NaN if no instalments"),
    "AVERAGE_PAYMENT_DELAY": ("mean(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)", "installments_payments", "Avg days late (negative = early)", "NaN if none"),
    "MAX_PAYMENT_DELAY": ("max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)", "installments_payments", "Worst lateness", "NaN if none"),
    "INSTALLMENT_PAYMENT_RATIO": ("sum(paid)/sum(due)", "installments_payments", "Paid vs due amount", "NaN if none"),
    "AVERAGE_CREDIT_UTILIZATION": ("mean(balance / limit)", "credit_card_balance", "Card utilisation", "NaN if no card or limit=0"),
    "MAX_CREDIT_UTILIZATION": ("max(balance / limit)", "credit_card_balance", "Peak card utilisation", "NaN if no card"),
}
LEAKAGE_FEATURE_NOTE = ("All history aggregates use only records with day-offsets <= 0 relative to the application "
                        "(DAYS_* / MONTHS_BALANCE). Future-dated values are neutralised or excluded in sql/*.sql.")


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").astype("float64"); b = pd.to_numeric(b, errors="coerce").astype("float64")
    out = a / b.where(b > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent: derives ratios from raw application columns when available."""
    df = df.copy()
    g = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    if "DAYS_EMPLOYED" in df.columns:
        anom = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        df["DAYS_EMPLOYED_ANOM"] = anom.astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(anom)
    df["CREDIT_INCOME_RATIO"] = safe_div(g("AMT_CREDIT"), g("AMT_INCOME_TOTAL"))
    df["ANNUITY_INCOME_RATIO"] = safe_div(g("AMT_ANNUITY"), g("AMT_INCOME_TOTAL"))
    df["CREDIT_ANNUITY_RATIO"] = safe_div(g("AMT_CREDIT"), g("AMT_ANNUITY"))
    df["INCOME_PER_PERSON"] = safe_div(g("AMT_INCOME_TOTAL"), g("CNT_FAM_MEMBERS"))
    df["EMPLOYMENT_DURATION_DAYS"] = -pd.to_numeric(g("DAYS_EMPLOYED"), errors="coerce")
    df["EMPLOYMENT_AGE_RATIO"] = safe_div(-pd.to_numeric(g("DAYS_EMPLOYED"), errors="coerce"),
                                          -pd.to_numeric(g("DAYS_BIRTH"), errors="coerce"))
    df["AGE_YEARS"] = -pd.to_numeric(g("DAYS_BIRTH"), errors="coerce") / 365.25
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    return df


def model_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    excl = {ID_COL, TARGET, *cfg["features"].get("exclude_from_model", [])}
    return [c for c in df.columns if c not in excl]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Drops columns that are too sparse, constant, or near-duplicate numerics. Fit on training data only."""
    def __init__(self, max_missing_frac=0.75, corr_threshold=0.98, corr_sample=50000, seed=42):
        self.max_missing_frac, self.corr_threshold, self.corr_sample, self.seed = max_missing_frac, corr_threshold, corr_sample, seed

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = [c for c in X.columns if X[c].isna().mean() <= self.max_missing_frac and X[c].nunique(dropna=True) > 1]
        num = X[keep].select_dtypes("number")
        if self.corr_threshold and num.shape[1] > 1:
            s = num.sample(min(len(num), self.corr_sample), random_state=self.seed)
            corr = s.corr().abs().to_numpy().copy(); np.fill_diagonal(corr, 0)
            cols = list(num.columns); drop = set()
            for i in range(len(cols)):
                if cols[i] in drop: continue
                for j in range(i + 1, len(cols)):
                    if cols[j] not in drop and corr[i, j] > self.corr_threshold:
                        drop.add(cols[j])
            keep = [c for c in keep if c not in drop]
        self.keep_ = keep; self.dropped_ = [c for c in X.columns if c not in keep]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


def make_preprocessor(kind: str, cfg: dict) -> Pipeline:
    """kind: 'linear' (impute+scale+one-hot) | 'tree_nan' (NaN-native: xgboost/lightgbm) | 'tree_imputed' (random forest)."""
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_exclude=np.number)
    if kind == "linear":
        num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=False)), ("sc", StandardScaler())])
        cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False))])
    elif kind == "tree_imputed":
        num = SimpleImputer(strategy="median")
        cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    else:
        num = "passthrough"
        cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    ct = ColumnTransformer([("num", num, num_sel), ("cat", cat, cat_sel)], verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    f = cfg["features"]
    return Pipeline([("select", FeatureSelector(f["max_missing_frac"], f["corr_prune_threshold"], seed=cfg["seed"])),
                     ("ct", ct)])


def _all_instances(values: pd.Series, types) -> bool:
    nn = values.dropna()
    return len(nn) > 0 and bool(nn.map(lambda v: isinstance(v, types)).all())


def clean_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Make every column either numeric or uniformly-typed strings, so imputers/encoders never see mixed types
    and training and inference match.
    - bool / nullable boolean / object with True,False(,None)  -> float 1.0 / 0.0 / NaN
    - object columns holding only numbers                       -> float
    - other object / string / category                          -> str (missing stays NaN)"""
    df = df.copy()
    for c in df.columns:
        s = df[c]; dt = str(s.dtype)
        if dt == "bool":
            df[c] = s.astype("float64")
        elif dt == "boolean":
            df[c] = s.astype("object").map(lambda v: np.nan if pd.isna(v) else float(bool(v))).astype("float64")
        elif dt == "object" or dt.startswith(("str", "category", "string")):
            o = s.astype("object")
            if _all_instances(o, (bool, np.bool_)):
                df[c] = o.map(lambda v: np.nan if pd.isna(v) else float(bool(v))).astype("float64")
            elif _all_instances(o, (int, float, np.integer, np.floating)):
                df[c] = pd.to_numeric(o, errors="coerce").astype("float64")
            else:
                df[c] = o.map(lambda v: np.nan if pd.isna(v) else str(v)).astype("object")
    return df

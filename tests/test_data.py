import numpy as np, pandas as pd
from creditiq import data_io


def test_required_columns_present(project):
    for t, cols in data_io.REQUIRED_COLUMNS.items():
        header = pd.read_csv(project.raw / f"{t}.csv", nrows=0).columns
        assert set(cols) <= set(header), t


def test_file_validation_passes(project):
    v = data_io.validate_files(project.raw)
    assert v.exists.all() and v.non_empty.all() and v.required_columns_ok.all()


def test_missing_file_fails(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError): data_io.validate_files(tmp_path)


def test_identifiers_and_target(project):
    tr = pd.read_parquet(project.processed / "features_train.parquet")
    te = pd.read_parquet(project.processed / "features_test.parquet")
    assert tr.SK_ID_CURR.is_unique and te.SK_ID_CURR.is_unique
    assert set(tr.TARGET.unique()) <= {0, 1}
    assert not (set(tr.SK_ID_CURR) & set(te.SK_ID_CURR))
    assert "TARGET" not in te.columns


def test_no_row_multiplication(project):
    raw = pd.read_csv(project.raw / "application_train.csv")
    tr = pd.read_parquet(project.processed / "features_train.parquet")
    assert len(raw) == len(tr)


def test_aggregation_matches_independent_pandas(project):
    prev = pd.read_csv(project.raw / "previous_application.csv"); prev = prev[prev.DAYS_DECISION <= 0]
    exp = prev.groupby("SK_ID_CURR").apply(lambda d: (d.NAME_CONTRACT_STATUS == "Approved").mean(), include_groups=False)
    tr = pd.read_parquet(project.processed / "features_train.parquet").set_index("SK_ID_CURR")
    common = exp.index.intersection(tr.index)
    assert len(common) > 100
    assert np.allclose(tr.loc[common, "PREVIOUS_APPROVAL_RATIO"].astype(float), exp.loc[common], atol=1e-6)
    b = pd.read_csv(project.raw / "bureau.csv"); cnt = b.groupby("SK_ID_CURR").size()
    c2 = cnt.index.intersection(tr.index)
    assert (tr.loc[c2, "BUREAU_PREV_CREDIT_COUNT"].astype(int) == cnt.loc[c2]).all()


def test_installment_delay_excludes_future_rows(project):
    inst = pd.read_csv(project.raw / "installments_payments.csv")
    ok = inst[(inst.DAYS_INSTALMENT <= 0) & (inst.DAYS_ENTRY_PAYMENT.fillna(0) <= 0)].copy()
    ok["late"] = (ok.DAYS_ENTRY_PAYMENT - ok.DAYS_INSTALMENT) > 0
    exp = ok.groupby("SK_ID_CURR").late.mean()
    tr = pd.read_parquet(project.processed / "features_train.parquet").set_index("SK_ID_CURR")
    c = exp.index.intersection(tr.index)
    assert np.allclose(tr.loc[c, "LATE_PAYMENT_RATIO"].astype(float), exp.loc[c], atol=1e-6)


def test_future_closing_date_neutralised(project):
    """30 injected DAYS_ENDDATE_FACT=500 rows must not reach the features."""
    tr = pd.read_parquet(project.processed / "features_train.parquet")
    assert (tr["BUREAU_DAYS_ENDDATE_FACT_MEAN"].dropna() <= 0).all()


def test_audit_outputs(project):
    a = pd.read_csv(project.reports / "dataset_audit.csv")
    assert len(a) == 8 and (a.rows > 0).all()
    rel = pd.read_csv(project.reports / "relationship_checks.csv")
    assert (rel.violations == 0).all()
    la = pd.read_csv(project.reports / "leakage_audit.csv")
    row = la[(la.table == "bureau") & (la.column == "DAYS_ENDDATE_FACT")].iloc[0]
    assert row.positive_values >= 30

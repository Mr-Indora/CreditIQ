"""Dataset audit, data dictionary, relationship checks and leakage/temporal audit."""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from .data_io import TABLES, find_raw_dir

TABLE_META = {  # primary key, foreign keys, cardinality, business meaning
    "application_train": ("SK_ID_CURR", "-", "1 row per applicant", "Loan application snapshot + TARGET (repayment difficulties label)"),
    "application_test": ("SK_ID_CURR", "-", "1 row per applicant", "Unlabelled applications (competition inference only)"),
    "bureau": ("SK_ID_BUREAU", "SK_ID_CURR -> application", "many per applicant", "Credits reported by other institutions"),
    "bureau_balance": ("(SK_ID_BUREAU, MONTHS_BALANCE)", "SK_ID_BUREAU -> bureau", "many per bureau credit", "Monthly status history of bureau credits"),
    "previous_application": ("SK_ID_PREV", "SK_ID_CURR -> application", "many per applicant", "Earlier Home Credit applications"),
    "POS_CASH_balance": ("(SK_ID_PREV, MONTHS_BALANCE)", "SK_ID_PREV -> previous_application; SK_ID_CURR", "many per previous loan", "Monthly POS/cash loan balance snapshots"),
    "installments_payments": ("-", "SK_ID_PREV -> previous_application; SK_ID_CURR", "many per previous loan", "Repayment history of previous loans"),
    "credit_card_balance": ("(SK_ID_PREV, MONTHS_BALANCE)", "SK_ID_PREV -> previous_application; SK_ID_CURR", "many per previous card", "Monthly credit-card balance snapshots"),
}
KEY_COLS = {"application_train": ["SK_ID_CURR"], "application_test": ["SK_ID_CURR"], "bureau": ["SK_ID_BUREAU"],
            "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"], "previous_application": ["SK_ID_PREV"],
            "POS_CASH_balance": ["SK_ID_PREV", "MONTHS_BALANCE"], "installments_payments": ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"],
            "credit_card_balance": ["SK_ID_PREV", "MONTHS_BALANCE"]}


def dataset_audit(paths) -> pd.DataFrame:
    """Per-table rows, columns, missing cells, duplicates on declared key (DuckDB, no full load)."""
    con = duckdb.connect(); rows = []
    for t in TABLES:
        f = (paths.interim / f"{t}.parquet").as_posix()
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()
        miss = con.execute("SELECT " + ", ".join(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols.column_name)
                           + f" FROM read_parquet('{f}')").fetchone()
        keys = [k for k in KEY_COLS[t] if k in set(cols.column_name)]
        dk = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT ({', '.join(keys)})) FROM read_parquet('{f}')").fetchone()[0] if keys else None
        pk, fk, card, meaning = TABLE_META[t]
        rows.append({"table": t, "rows": n, "columns": len(cols), "missing_cells": int(sum(m or 0 for m in miss)),
                     "missing_pct": round(100 * sum(m or 0 for m in miss) / max(1, n * len(cols)), 2),
                     "duplicate_key_rows": dk, "primary_key": pk, "foreign_keys": fk, "cardinality": card, "business_meaning": meaning})
    con.close()
    out = pd.DataFrame(rows); out.to_csv(paths.reports / "dataset_audit.csv", index=False)
    return out


def relationship_checks(paths) -> pd.DataFrame:
    """Referential-integrity checks between tables (orphans, cardinality)."""
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(paths.interim / (t + '.parquet')).as_posix()}')")
    q = {
        "bureau.SK_ID_CURR in application": "SELECT COUNT(*) FROM bureau b WHERE b.SK_ID_CURR NOT IN (SELECT SK_ID_CURR FROM application_train UNION SELECT SK_ID_CURR FROM application_test)",
        "bureau_balance.SK_ID_BUREAU in bureau": "SELECT COUNT(*) FROM bureau_balance x WHERE x.SK_ID_BUREAU NOT IN (SELECT SK_ID_BUREAU FROM bureau)",
        "previous_application.SK_ID_CURR in application": "SELECT COUNT(*) FROM previous_application p WHERE p.SK_ID_CURR NOT IN (SELECT SK_ID_CURR FROM application_train UNION SELECT SK_ID_CURR FROM application_test)",
        "POS.SK_ID_PREV in previous_application": "SELECT COUNT(*) FROM POS_CASH_balance x WHERE x.SK_ID_PREV NOT IN (SELECT SK_ID_PREV FROM previous_application)",
        "installments.SK_ID_PREV in previous_application": "SELECT COUNT(*) FROM installments_payments x WHERE x.SK_ID_PREV NOT IN (SELECT SK_ID_PREV FROM previous_application)",
        "credit_card.SK_ID_PREV in previous_application": "SELECT COUNT(*) FROM credit_card_balance x WHERE x.SK_ID_PREV NOT IN (SELECT SK_ID_PREV FROM previous_application)",
        "train/test applicant overlap": "SELECT COUNT(*) FROM application_train a JOIN application_test t USING (SK_ID_CURR)",
    }
    rows = [{"check": k, "violations": con.execute(v).fetchone()[0]} for k, v in q.items()]
    con.close()
    out = pd.DataFrame(rows); out.to_csv(paths.reports / "relationship_checks.csv", index=False)
    return out


def build_data_dictionary(paths) -> pd.DataFrame:
    """Column-level dictionary from actual data; merges official descriptions if HomeCredit_columns_description.csv exists."""
    con = duckdb.connect(); parts = []
    for t in TABLES:
        f = (paths.interim / f"{t}.parquet").as_posix()
        d = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()[["column_name", "column_type"]]
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]
        nulls = con.execute("SELECT " + ", ".join(f'AVG(CASE WHEN "{c}" IS NULL THEN 1.0 ELSE 0.0 END)' for c in d.column_name)
                            + f" FROM read_parquet('{f}')").fetchone()
        d["table"] = t; d["missing_pct"] = [round(100 * (x or 0), 2) for x in nulls]; d["rows"] = n
        parts.append(d)
    con.close()
    dd = pd.concat(parts, ignore_index=True)
    raw = find_raw_dir(paths); desc_f = (raw / "HomeCredit_columns_description.csv") if raw else None
    dd["description"] = ""
    if desc_f is not None and desc_f.exists():
        desc = pd.read_csv(desc_f, encoding="latin-1")
        desc["Table"] = desc["Table"].str.replace(".csv", "", regex=False)
        m = {(r.Table.lower(), r.Row): r.Description for r in desc.itertuples()}
        dd["description"] = [m.get((t.lower(), c), "") for t, c in zip(dd["table"], dd.column_name)]
    dd.to_csv(paths.reports / "data_dictionary.csv", index=False)
    return dd


# ---------------- leakage / temporal audit ----------------
def leakage_audit(paths) -> pd.DataFrame:
    """Checks the sign of every relative-time column (DAYS_*, MONTHS_BALANCE). Convention: <= 0 means before application.
    Positive values are either planned future dates (known at application, e.g. contractual end) or post-application facts (unsafe).
    The 365243 sentinel is reported separately. This audit CANNOT prove temporal validity - it checks consistency with the documented convention."""
    con = duckdb.connect(); rows = []
    for t in TABLES:
        f = (paths.interim / f"{t}.parquet").as_posix()
        d = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()
        for c in d.column_name:
            if c.startswith("DAYS_") or c == "MONTHS_BALANCE":
                r = con.execute(f'SELECT COUNT("{c}"), SUM(CASE WHEN "{c}" = 365243 THEN 1 ELSE 0 END), '
                                f'SUM(CASE WHEN "{c}" > 0 AND "{c}" <> 365243 THEN 1 ELSE 0 END) FROM read_parquet(\'{f}\')').fetchone()
                n, sent, pos = r[0], r[1] or 0, r[2] or 0
                rows.append({"table": t, "column": c, "non_null": n, "sentinel_365243": sent, "positive_values": pos,
                             "positive_share": round(pos / max(1, n), 6), "treatment": _treatment(t, c, pos, sent)})
    con.close()
    out = pd.DataFrame(rows)
    out.to_csv(paths.reports / "leakage_audit.csv", index=False)
    return out


def _treatment(t, c, pos, sent):
    if t == "bureau" and c == "DAYS_ENDDATE_FACT":
        return "Positive values nulled in sql/bureau.sql (actual closing date after application = post-application info)"
    if t == "bureau" and c == "DAYS_CREDIT_ENDDATE":
        return "Planned end date, contractually known at application -> not used directly (excluded from aggregation)"
    if c == "DAYS_EMPLOYED" and sent:
        return "Sentinel 365243 -> NaN + DAYS_EMPLOYED_ANOM flag"
    if pos:
        return "Rows with positive offset filtered in sql/*.sql"
    return "Consistent with convention (<= 0); not independently verifiable"


def feature_availability_report(feature_cols: list[str], cfg: dict) -> pd.DataFrame:
    """Per-feature availability documentation: source + whether in primary model."""
    prefix_src = {"BUREAU_": "bureau/bureau_balance", "PREV": "previous_application", "POS_": "POS_CASH_balance",
                  "INST": "installments_payments", "CC_": "credit_card_balance", "HAS_": "derived flag"}
    rows = []
    for c in feature_cols:
        src = next((v for k, v in prefix_src.items() if c.startswith(k)), None)
        if src is None:
            src = {"ACTIVE_CREDIT_RATIO": "bureau", "AVERAGE_CREDIT_DEBT": "bureau", "OVERDUE_ACCOUNT_RATIO": "bureau",
                   "LATE_PAYMENT_RATIO": "installments_payments", "AVERAGE_PAYMENT_DELAY": "installments_payments",
                   "MAX_PAYMENT_DELAY": "installments_payments", "INSTALLMENT_PAYMENT_RATIO": "installments_payments",
                   "AVERAGE_CREDIT_UTILIZATION": "credit_card_balance", "MAX_CREDIT_UTILIZATION": "credit_card_balance",
                   "PREVIOUS_APPROVAL_RATIO": "previous_application"}.get(c, "application")
        rows.append({"feature": c, "source": src, "available_at_application": "yes (per dataset day-offset convention; not independently verifiable)" if src != "application" else "yes (applicant-provided/at application)"})
    return pd.DataFrame(rows)

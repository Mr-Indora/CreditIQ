"""Relational aggregation with DuckDB SQL (streaming, memory-safe) -> applicant-level feature table."""
from __future__ import annotations
import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from .data_io import TABLES
from .features import add_application_features

AGG_SQL = {"bureau": "bureau.sql", "previous_application": "previous_application.sql",
           "pos_cash": "pos_cash.sql", "installments": "installments.sql", "credit_card": "credit_card.sql"}


def _connect(paths) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        f = paths.interim / f"{t}.parquet"
        if f.exists():
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    return con


def run_aggregations(paths, force: bool = False) -> dict[str, Path]:
    """Run each SQL file; checkpoint each result as parquet (resumable)."""
    con = _connect(paths); out = {}
    for name, sqlfile in AGG_SQL.items():
        dst = paths.interim / f"agg_{name}.parquet"
        if dst.exists() and not force:
            out[name] = dst; continue
        sql = (paths.sql / sqlfile).read_text()
        df = con.execute(sql).df()
        assert df.SK_ID_CURR.is_unique, f"{name}: aggregation not unique per SK_ID_CURR"
        df.to_parquet(dst, index=False)
        out[name] = dst
    con.close()
    return out


def run_analysis_queries(paths) -> dict[str, pd.DataFrame]:
    """Execute sql/analysis_queries.sql blocks ('-- name: x' + statement) and save CSVs."""
    con = _connect(paths); res = {}
    txt = (paths.sql / "analysis_queries.sql").read_text()
    for m in re.finditer(r"-- name: (\w+)\n(.*?;)", txt, re.S):
        name, q = m.group(1), m.group(2)
        res[name] = con.execute(q).df()
        res[name].to_csv(paths.reports / f"sql_{name}.csv", index=False)
    con.close()
    return res


def build_feature_table(paths, split: str = "train") -> pd.DataFrame:
    """One row per SK_ID_CURR: application + aggregated history + engineered ratios."""
    app = pd.read_parquet(paths.interim / f"application_{split}.parquet")
    n0 = len(app)
    assert app.SK_ID_CURR.is_unique, "application table has duplicate SK_ID_CURR"
    df = app
    for name in AGG_SQL:
        agg = pd.read_parquet(paths.interim / f"agg_{name}.parquet")
        df = df.merge(agg, on="SK_ID_CURR", how="left", validate="one_to_one")
        flag = {"bureau": "HAS_BUREAU", "previous_application": "HAS_PREV_APP", "pos_cash": "HAS_POS",
                "installments": "HAS_INSTALLMENTS", "credit_card": "HAS_CREDIT_CARD"}[name]
        df[flag] = df[[c for c in agg.columns if c != "SK_ID_CURR"][0]].notna().astype("int8")
    assert len(df) == n0, "row multiplication detected after merge"
    count_cols = [c for c in df.columns if c.endswith("_COUNT")]
    df[count_cols] = df[count_cols].fillna(0)   # no history => zero events (documented)
    df = add_application_features(df)
    df = downcast(df)
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        if c not in ("SK_ID_CURR",):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

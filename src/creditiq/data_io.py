"""Dataset acquisition (Kaggle API or manual upload), validation, CSV->Parquet conversion."""
from __future__ import annotations
import os, subprocess, zipfile
from pathlib import Path
import duckdb
import pandas as pd

COMPETITION = "home-credit-default-risk"
TABLES = ["application_train", "application_test", "bureau", "bureau_balance",
          "previous_application", "POS_CASH_balance", "installments_payments", "credit_card_balance"]

# columns the pipeline relies on (schema validation)
REQUIRED_COLUMNS = {
    "application_train": ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                          "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER"],
    "application_test": ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
               "DAYS_ENDDATE_FACT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
               "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    "previous_application": ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
                             "AMT_CREDIT", "DAYS_DECISION", "CNT_PAYMENT"],
    "POS_CASH_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF",
                         "NAME_CONTRACT_STATUS"],
    "installments_payments": ["SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                              "AMT_INSTALMENT", "AMT_PAYMENT"],
    "credit_card_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                            "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_CURRENT", "SK_DPD"],
}


def find_raw_dir(paths) -> Path | None:
    """Locate a directory containing the competition CSVs (raw dir, or one subfolder below)."""
    for cand in [paths.raw, *[p for p in paths.raw.glob("*") if p.is_dir()]]:
        if (cand / "application_train.csv").exists():
            return cand
    return None


def get_kaggle_credentials() -> bool:
    """Read credentials from Colab Secrets (KAGGLE_USERNAME / KAGGLE_KEY) or env. Never hardcode."""
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        userdata = None
    if userdata is not None:
        for key in ("KAGGLE_API_TOKEN",):          # new-style access token (optional secret)
            try: os.environ[key] = userdata.get(key); return True
            except Exception: pass
        try:                                        # classic username + key
            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
            return True
        except Exception:
            pass
    return (Path.home() / ".kaggle" / "kaggle.json").exists() or (Path.home() / ".kaggle" / "access_token").exists()


def download_dataset(paths) -> Path | None:
    """Kaggle API download. Requires accepting the competition rules on Kaggle first."""
    existing = find_raw_dir(paths)
    if existing:
        print(f"Raw data already present at {existing}; skipping download.")
        return existing
    if not get_kaggle_credentials():
        print(MANUAL_INSTRUCTIONS)
        return None
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(paths.raw)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Kaggle download failed:\n", r.stderr[-800:], "\n", MANUAL_INSTRUCTIONS)
        return None
    for z in paths.raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(paths.raw)
    return find_raw_dir(paths)


MANUAL_INSTRUCTIONS = """
Kaggle credentials not available / download failed. Options:
 1. Add Colab Secrets KAGGLE_USERNAME and KAGGLE_KEY (Colab left sidebar -> key icon), accept the
    competition rules at https://www.kaggle.com/competitions/home-credit-default-risk/rules, re-run.
 2. Manual: download the zip from https://www.kaggle.com/competitions/home-credit-default-risk/data,
    upload the 8 CSVs (and optionally HomeCredit_columns_description.csv) to MyDrive/CreditIQ/data/raw/
"""


def validate_files(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for t in TABLES:
        f = raw_dir / f"{t}.csv"
        ok = f.exists() and f.stat().st_size > 0
        cols_ok, missing = False, []
        if ok:
            header = pd.read_csv(f, nrows=0).columns.tolist()
            missing = [c for c in REQUIRED_COLUMNS[t] if c not in header]
            cols_ok = not missing
        rows.append({"table": t, "exists": f.exists(), "non_empty": ok,
                     "required_columns_ok": cols_ok, "missing_columns": missing})
    df = pd.DataFrame(rows)
    if not (df.exists & df.non_empty & df.required_columns_ok).all():
        raise FileNotFoundError(f"Dataset validation failed:\n{df.to_string()}")
    return df


def csv_to_parquet(paths, raw_dir: Path, force: bool = False) -> dict[str, Path]:
    """Convert CSVs to Parquet with DuckDB (streaming, low memory). Resumable per table."""
    con = duckdb.connect()
    out = {}
    for t in TABLES:
        dst = paths.interim / f"{t}.parquet"
        if dst.exists() and not force:
            out[t] = dst
            continue
        src = (raw_dir / f"{t}.csv").as_posix()
        con.execute(f"COPY (SELECT * FROM read_csv_auto('{src}', sample_size=-1, header=true)) "
                    f"TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        out[t] = dst
    con.close()
    return out


def load_parquet(paths, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(paths.interim / f"{table}.parquet", columns=columns)

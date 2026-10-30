"""Synthetic data in the Home Credit schema.

FOR SMOKE TESTS / CI ONLY. Metrics produced from this data are meaningless for any real claim.
The generator writes the same 8 CSV files so that the full pipeline is exercised end to end.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic(raw_dir: Path, n_train=4000, n_test=500, seed=0) -> None:
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    n = n_train + n_test
    ids = np.arange(100001, 100001 + n)
    income = np.exp(rng.normal(11.9, 0.5, n)); credit = income * rng.uniform(1, 6, n)
    annuity = credit / rng.uniform(10, 40, n)
    ext = rng.uniform(0, 1, (n, 3)); ext[rng.random((n, 3)) < 0.2] = np.nan
    days_birth = -rng.integers(7500, 24000, n)
    days_emp = -rng.integers(30, 12000, n).astype(float)
    days_emp[rng.random(n) < 0.15] = 365243
    gender = rng.choice(["F", "M"], n, p=[.65, .35])
    # latent risk drives target and some history features
    lat = (-2.2 - 1.8 * np.nan_to_num(ext[:, 1], nan=.5) + 0.35 * np.log(credit / income)
           + 0.0000 * days_birth + rng.normal(0, .6, n))
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": (rng.random(n) < 1 / (1 + np.exp(-lat))).astype(int),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[.9, .1]),
        "CODE_GENDER": gender, "FLAG_OWN_CAR": rng.choice(["Y", "N"], n), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "CNT_CHILDREN": rng.integers(0, 4, n), "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": credit * rng.uniform(.8, 1, n),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], n),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment"], n),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", "Sales staff", None], n),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "School", "Government", "XNA"], n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n), "REGION_RATING_CLIENT": rng.integers(1, 4, n),
        "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_emp,
        "EXT_SOURCE_1": ext[:, 0], "EXT_SOURCE_2": ext[:, 1], "EXT_SOURCE_3": ext[:, 2],
    })
    app.iloc[:n_train].to_csv(raw_dir / "application_train.csv", index=False)
    app.iloc[n_train:].drop(columns="TARGET").to_csv(raw_dir / "application_test.csv", index=False)

    # bureau (~70% of applicants have it)
    has = rng.random(n) < .7
    nb = np.where(has, rng.integers(1, 8, n), 0)
    b_cur = np.repeat(ids, nb); m = len(b_cur)
    risk_rep = np.repeat(lat, nb)
    dcred = -rng.integers(30, 3000, m)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_cur, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold"], m, p=[.4, .55, .05]),
        "CREDIT_CURRENCY": "currency 1", "DAYS_CREDIT": dcred,
        "CREDIT_DAY_OVERDUE": np.where(rng.random(m) < .05 + .05 * (risk_rep > -1), rng.integers(1, 90, m), 0),
        "DAYS_CREDIT_ENDDATE": dcred + rng.integers(100, 2000, m),
        "DAYS_ENDDATE_FACT": np.where(rng.random(m) < .5, dcred + rng.integers(50, 1500, m), np.nan),
        "AMT_CREDIT_MAX_OVERDUE": np.where(rng.random(m) < .2, rng.uniform(0, 5000, m), np.nan),
        "CNT_CREDIT_PROLONG": rng.integers(0, 2, m),
        "AMT_CREDIT_SUM": np.exp(rng.normal(11, 1, m)), "AMT_CREDIT_SUM_DEBT": np.exp(rng.normal(9.5, 1.5, m)),
        "AMT_CREDIT_SUM_LIMIT": rng.uniform(0, 1e4, m), "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(m) < .04, rng.uniform(1, 2000, m), 0),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], m),
        "DAYS_CREDIT_UPDATE": -rng.integers(1, 500, m), "AMT_ANNUITY": np.nan})
    # inject deliberate future value (must be neutralised by the pipeline)
    bureau.loc[bureau.sample(30, random_state=1).index, "DAYS_ENDDATE_FACT"] = 500
    bureau.to_csv(raw_dir / "bureau.csv", index=False)
    nbb = rng.integers(1, 12, m)
    bb = pd.DataFrame({"SK_ID_BUREAU": np.repeat(bureau.SK_ID_BUREAU.values, nbb)})
    bb["MONTHS_BALANCE"] = -np.concatenate([np.arange(k) for k in nbb])
    bb["STATUS"] = rng.choice(["C", "X", "0", "1", "2", "3"], len(bb), p=[.3, .1, .45, .08, .04, .03])
    bb.to_csv(raw_dir / "bureau_balance.csv", index=False)

    # previous applications
    npv = rng.integers(0, 6, n); p_cur = np.repeat(ids, npv); mp = len(p_cur)
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(2000000, 2000000 + mp), "SK_ID_CURR": p_cur,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans"], mp),
        "AMT_ANNUITY": rng.uniform(1e3, 3e4, mp), "AMT_APPLICATION": np.exp(rng.normal(11, 1, mp)),
        "AMT_CREDIT": np.exp(rng.normal(11, 1, mp)), "AMT_DOWN_PAYMENT": np.nan, "AMT_GOODS_PRICE": np.nan,
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled", "Unused offer"], mp, p=[.65, .17, .17, .01]),
        "DAYS_DECISION": -rng.integers(1, 2900, mp), "CNT_PAYMENT": rng.choice([6, 12, 24, 36, np.nan], mp)})
    prev.to_csv(raw_dir / "previous_application.csv", index=False)

    ap = prev[prev.NAME_CONTRACT_STATUS == "Approved"]
    # POS
    k = rng.integers(1, 15, len(ap))
    pos = pd.DataFrame({"SK_ID_PREV": np.repeat(ap.SK_ID_PREV.values, k), "SK_ID_CURR": np.repeat(ap.SK_ID_CURR.values, k)})
    pos["MONTHS_BALANCE"] = -np.concatenate([np.arange(x) for x in k])
    pos["CNT_INSTALMENT"] = 12.0; pos["CNT_INSTALMENT_FUTURE"] = 6.0
    pos["NAME_CONTRACT_STATUS"] = rng.choice(["Active", "Completed"], len(pos), p=[.9, .1])
    pos["SK_DPD"] = np.where(rng.random(len(pos)) < .05, rng.integers(1, 100, len(pos)), 0)
    pos["SK_DPD_DEF"] = np.where(pos.SK_DPD > 20, pos.SK_DPD, 0)
    pos.to_csv(raw_dir / "POS_CASH_balance.csv", index=False)
    # installments
    k = rng.integers(1, 10, len(ap))
    inst = pd.DataFrame({"SK_ID_PREV": np.repeat(ap.SK_ID_PREV.values, k), "SK_ID_CURR": np.repeat(ap.SK_ID_CURR.values, k)})
    inst["NUM_INSTALMENT_VERSION"] = 1; inst["NUM_INSTALMENT_NUMBER"] = np.concatenate([np.arange(1, x + 1) for x in k])
    inst["DAYS_INSTALMENT"] = -rng.integers(1, 2500, len(inst)).astype(float)
    inst["DAYS_ENTRY_PAYMENT"] = inst.DAYS_INSTALMENT + rng.integers(-5, 15, len(inst))
    inst.loc[inst.DAYS_ENTRY_PAYMENT > 0, "DAYS_ENTRY_PAYMENT"] = -1.0
    inst["AMT_INSTALMENT"] = rng.uniform(1e3, 2e4, len(inst))
    inst["AMT_PAYMENT"] = inst.AMT_INSTALMENT * rng.choice([1, 1, 1, .5], len(inst))
    inst.to_csv(raw_dir / "installments_payments.csv", index=False)
    # credit card
    cc_ids = ap.sample(frac=.3, random_state=2); k = rng.integers(1, 12, len(cc_ids))
    cc = pd.DataFrame({"SK_ID_PREV": np.repeat(cc_ids.SK_ID_PREV.values, k),
                       "SK_ID_CURR": np.repeat(cc_ids.SK_ID_CURR.values, k)})
    cc["MONTHS_BALANCE"] = -np.concatenate([np.arange(x) for x in k])
    cc["AMT_CREDIT_LIMIT_ACTUAL"] = rng.choice([45000, 90000, 135000, 0], len(cc), p=[.3, .3, .35, .05])
    cc["AMT_BALANCE"] = cc.AMT_CREDIT_LIMIT_ACTUAL * rng.uniform(0, 1.1, len(cc))
    cc["AMT_DRAWINGS_CURRENT"] = rng.uniform(0, 2e4, len(cc)); cc["AMT_PAYMENT_CURRENT"] = rng.uniform(0, 2e4, len(cc))
    cc["AMT_TOTAL_RECEIVABLE"] = cc.AMT_BALANCE
    cc["SK_DPD"] = np.where(rng.random(len(cc)) < .03, rng.integers(1, 60, len(cc)), 0); cc["SK_DPD_DEF"] = 0
    cc["NAME_CONTRACT_STATUS"] = "Active"
    cc.to_csv(raw_dir / "credit_card_balance.csv", index=False)

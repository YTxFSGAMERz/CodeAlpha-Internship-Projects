"""
Feature engineering and preprocessing for the credit-scoring pipeline.

`engineer_features` turns raw financial history into behaviour-based signals
(credit utilisation, how much of each bill is repaid, how late and how often ...).
It lives in its own module so a saved model can be loaded from any script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Lenders in many countries may not use these for credit decisions (e.g. ECOA in the US),
# so they are removed by default. Use `train.py --keep-protected` to see their effect.
PROTECTED_COLUMNS = ["SEX", "MARRIAGE"]
EDUCATION_LABELS = {1: "graduate_school", 2: "university", 3: "high_school"}


def _engineer_card_history(X: pd.DataFrame) -> pd.DataFrame:
    """Features for the UCI credit-card data (6 months of bills, payments and repayment status)."""
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]       # BILL_AMT1 = most recent month
    paid = [f"PAY_AMT{i}" for i in range(1, 7)]
    status = [f"PAY_{i}" for i in range(1, 7)]         # -2/-1 paid duly, 0 revolving, 1..9 months late
    limit = X["LIMIT_BAL"].where(X["LIMIT_BAL"] > 0)   # NaN instead of dividing by zero

    # how much of the credit line is being used
    X["avg_bill"] = X[bill].mean(axis=1)
    X["utilization_latest"] = (X["BILL_AMT1"] / limit).clip(-1, 3)
    X["utilization_avg"] = (X["avg_bill"] / limit).clip(-1, 3)
    X["utilization_peak"] = (X[bill].max(axis=1) / limit).clip(-1, 3)

    # how much of what is owed actually gets paid
    total_bill = X[bill].clip(lower=0).sum(axis=1)
    X["avg_payment"] = X[paid].mean(axis=1)
    X["payment_ratio"] = (X[paid].sum(axis=1) / total_bill.where(total_bill > 0)).clip(0, 5)

    # repayment behaviour
    delays = X[status].clip(lower=0)                    # only real delays count
    X["max_delay"] = delays.max(axis=1)
    X["avg_delay"] = delays.mean(axis=1)
    X["months_late"] = (X[status] > 0).sum(axis=1)
    X["recent_delay"] = delays[["PAY_1", "PAY_2", "PAY_3"]].sum(axis=1)   # last 3 months
    X["delay_trend"] = X["PAY_1"] - X["PAY_6"]          # > 0 means getting worse

    # is the debt growing?
    X["bill_trend"] = ((X["BILL_AMT1"] - X["BILL_AMT6"]) / limit).clip(-3, 3)

    # readable categories, so they are one-hot encoded instead of read as magnitudes
    if "EDUCATION" in X:
        X["EDUCATION"] = X["EDUCATION"].map(EDUCATION_LABELS).fillna("other")
    if "SEX" in X:
        X["SEX"] = X["SEX"].map({1: "male", 2: "female"})
    if "MARRIAGE" in X:
        X["MARRIAGE"] = X["MARRIAGE"].map({1: "married", 2: "single"}).fillna("other")
    return X


def _engineer_give_me_some_credit(X: pd.DataFrame) -> pd.DataFrame:
    """Features for the Kaggle 'Give Me Some Credit' file (income, debt ratio, late payments)."""
    late = [c for c in X.columns if "PastDue" in c or "DaysLate" in c]
    for c in late:                                     # 96 / 98 are junk codes in this file
        X[c] = X[c].where(X[c] < 90)
    X["total_late_payments"] = X[late].sum(axis=1)
    X["income_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["log_income"] = np.log1p(X["MonthlyIncome"].clip(lower=0))
    X["DebtRatio"] = X["DebtRatio"].clip(upper=10)
    X["monthly_debt"] = X["DebtRatio"] * X["MonthlyIncome"]
    X["RevolvingUtilizationOfUnsecuredLines"] = X["RevolvingUtilizationOfUnsecuredLines"].clip(upper=2)
    if "age" in X:
        X["age"] = X["age"].where(X["age"] >= 18)
    return X


def engineer_features(df: pd.DataFrame, drop_protected: bool = True) -> pd.DataFrame:
    """Add engineered columns (only for the column layouts it recognises) and tidy the data."""
    X = df.copy()
    cols = set(X.columns)
    if {"LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "PAY_1", "PAY_6"} <= cols:
        X = _engineer_card_history(X)
    if {"MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"} <= cols:
        X = _engineer_give_me_some_credit(X)
    if drop_protected:
        X = X.drop(columns=[c for c in PROTECTED_COLUMNS if c in X.columns])
    return X.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> ColumnTransformer:
    """Numeric columns: median-impute + standardise. Text columns: mode-impute + one-hot."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", numeric, make_column_selector(dtype_include=np.number)),
         ("cat", categorical, make_column_selector(dtype_exclude=np.number))],
        sparse_threshold=0)


def build_pipeline(estimator, drop_protected: bool = True) -> Pipeline:
    """raw data -> feature engineering -> preprocessing -> classifier, as ONE object."""
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, kw_args={"drop_protected": drop_protected})),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])

"""
Data loading for the Credit Scoring project.

Every loader returns  (X, y):
    X  pandas DataFrame with the raw applicant features
    y  pandas Series, 1 = the customer defaulted ("bad credit"), 0 = repaid ("good credit")

Sources
-------
credit_default  "Default of Credit Card Clients" (UCI Machine Learning Repository, Taiwan).
                24,000 clients: credit limit, 6 months of repayment status, bill amounts and
                payment amounts, and whether the client defaulted the following month.
                A copy is bundled in ./data (it is downloaded once if the file is missing).
csv             Any CSV file with a binary target column, for example the Kaggle
                "Give Me Some Credit" file:
                    python train.py --source csv --data cs-training.csv --target SeriousDlqin2yrs
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CREDIT_DEFAULT_PATH = DATA_DIR / "credit_default.csv"
CREDIT_DEFAULT_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv"


def _normalise_credit_default(df: pd.DataFrame) -> pd.DataFrame:
    """The UCI, Kaggle and mirror versions use slightly different headers - unify them."""
    df = df.rename(columns={"PAY_0": "PAY_1",
                            "default.payment.next.month": "default",
                            "default payment next month": "default"})
    return df.drop(columns=[c for c in ("ID",) if c in df.columns])


def load_credit_default():
    if not CREDIT_DEFAULT_PATH.exists():
        print("Downloading the credit-default dataset (one time only) ...")
        DATA_DIR.mkdir(exist_ok=True)
        pd.read_csv(CREDIT_DEFAULT_URL).to_csv(CREDIT_DEFAULT_PATH, index=False)
    df = _normalise_credit_default(pd.read_csv(CREDIT_DEFAULT_PATH))
    y = df.pop("default").astype(int)
    return df, y


def load_csv(path, target, positive_label=None):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])  # saved index
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    df = df[df[target].notna()]
    y = df.pop(target)
    if pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        y = y.astype(int)
    else:                                   # text labels such as good / bad
        if positive_label is None:
            raise ValueError("The target is text - say which value means default, "
                             "e.g. --positive-label bad")
        y = (y.astype(str).str.strip().str.lower() == str(positive_label).strip().lower()).astype(int)
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"Target must be binary (0/1) but contains {sorted(y.unique())}")
    return df, y


def load_dataset(source="credit_default", data=None, target=None, positive_label=None):
    if source == "credit_default":
        return load_credit_default()
    if source == "csv":
        if not data or not target:
            raise ValueError("--source csv needs both --data <file.csv> and --target <column>")
        return load_csv(data, target, positive_label)
    raise ValueError(f"Unknown source '{source}' (use credit_default or csv)")

"""
Score customers with the trained credit model.

    python predict.py --sample 8                      # 8 random customers from the training data
    python predict.py --csv new_applicants.csv        # a CSV with the same raw columns
    python predict.py --json '{"LIMIT_BAL": 50000, "AGE": 35, "PAY_1": 2, "BILL_AMT1": 48000, ...}'

Missing columns are imputed, so partial applications still get a score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data_loader import load_dataset
from scorecard import decision, probability_to_score

MODEL_PATH = Path(__file__).resolve().parent / "models" / "credit_model.joblib"


def main():
    ap = argparse.ArgumentParser(description="Score customers with the trained credit model.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, metavar="N", help="score N random rows of the dataset")
    group.add_argument("--csv", help="CSV file with applicants (raw columns)")
    group.add_argument("--json", help="a single applicant as a JSON string")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run  python train.py  first.")
    bundle = joblib.load(MODEL_PATH)

    actual = None
    if args.sample:
        X, y = load_dataset(bundle["source"], bundle["data"], bundle["target"], bundle["positive_label"])
        rows = X.sample(args.sample, random_state=args.seed)
        actual = y.loc[rows.index]
    elif args.csv:
        rows = pd.read_csv(args.csv).rename(columns={"PAY_0": "PAY_1"})
    else:
        rows = pd.DataFrame([json.loads(args.json)]).rename(columns={"PAY_0": "PAY_1"})

    missing = [c for c in bundle["input_columns"] if c not in rows.columns]
    if missing:
        print(f"Note: {len(missing)} column(s) not provided and will be imputed: {missing[:6]}"
              f"{' ...' if len(missing) > 6 else ''}\n")
    rows = rows.reindex(columns=bundle["input_columns"])

    proba = bundle["pipeline"].predict_proba(rows)[:, 1]
    result = pd.DataFrame({
        "default_probability": [f"{p:.1%}" for p in proba],
        "credit_score": probability_to_score(proba),
        "decision": decision(proba, bundle["threshold"]),
    }, index=rows.index)
    if actual is not None:
        result["actually_defaulted"] = actual.map({1: "yes", 0: "no"})

    print(f"Model: {bundle['model_name']}  |  decline threshold: {bundle['threshold']:.0%} default probability\n")
    print(result.to_string())


if __name__ == "__main__":
    main()

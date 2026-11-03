"""
Predict disease risk for ONE patient with a model trained by train.py.

    python predict.py --dataset heart --describe          # list the expected inputs
    python predict.py --dataset heart --example           # use a real row from the dataset
    python predict.py --dataset heart --json '{"age": 58, "sex": 1, "cp": 4, ...}'
    python predict.py --dataset diabetes --file patient.json

Educational project - NOT a medical device and NOT medical advice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data_loader import DATASETS, FEATURE_HELP, clean_features, load_dataset

MODEL_DIR = Path(__file__).resolve().parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Predict disease risk for a single patient.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", help="patient values as a JSON string")
    group.add_argument("--file", help="path to a JSON file with the patient values")
    group.add_argument("--example", action="store_true", help="score a random real patient from the dataset")
    group.add_argument("--describe", action="store_true", help="show what each input means and exit")
    args = parser.parse_args()

    if args.describe:
        help_text = FEATURE_HELP.get(args.dataset)
        if help_text is None:
            X, _, _ = load_dataset(args.dataset)
            help_text = {c: "numeric measurement" for c in X.columns}
        for feature, text in help_text.items():
            print(f"  {feature:<26} {text}")
        return

    model_path = MODEL_DIR / f"{args.dataset}_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"No trained model found at {model_path}. Run:  python train.py --dataset {args.dataset}")
    bundle = joblib.load(model_path)

    actual = None
    if args.example:
        X, y, _ = load_dataset(args.dataset)
        idx = X.sample(1).index[0]
        patient, actual = X.loc[[idx]], int(y.loc[idx])
        print("Example patient:\n", patient.iloc[0].to_string(), "\n")
    elif args.json or args.file:
        values = json.loads(args.json) if args.json else json.loads(Path(args.file).read_text())
        patient = clean_features(args.dataset, pd.DataFrame([values]))
        missing = [c for c in bundle["features"] if c not in patient.columns]
        if missing:
            print(f"Warning: missing values for {missing} - they will be imputed.")
        patient = patient.reindex(columns=bundle["features"])
    else:
        parser.error("give one of --json, --file, --example or --describe")

    proba = float(bundle["pipeline"].predict_proba(patient)[0, 1])
    label = "DISEASE LIKELY" if proba >= bundle["threshold"] else "disease unlikely"
    print(f"Model     : {bundle['model_name']}  ({bundle['title']})")
    print(f"Risk score: {proba:.1%} probability of disease  ->  {label}")
    if actual is not None:
        print(f"Actual    : {'disease present' if actual else 'no disease'} (from the dataset)")
    print("\nEducational project only - this is not medical advice. See a doctor for health concerns.")


if __name__ == "__main__":
    main()

"""
Automated unit tests for Disease Prediction Model.
"""
import sys
from pathlib import Path
import pytest
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import load_dataset, DATASETS


def test_datasets_load():
    for name in DATASETS:
        X, y, info = load_dataset(name)
        assert len(X) > 0, f"Dataset {name} is empty"
        assert len(y) == len(X)
        assert set(y.unique()) <= {0, 1}
        assert "title" in info


def test_saved_models_inference():
    model_dir = ROOT / "models"
    for name in ["heart", "diabetes", "breast_cancer"]:
        model_file = model_dir / f"{name}_model.joblib"
        if model_file.exists():
            bundle = joblib.load(model_file)
            assert "pipeline" in bundle
            assert "model_name" in bundle
            pipeline = bundle["pipeline"]
            X, y, _ = load_dataset(name)
            sample = X.iloc[:5]
            probs = pipeline.predict_proba(sample)[:, 1]
            assert len(probs) == 5
            assert all(0.0 <= p <= 1.0 for p in probs)

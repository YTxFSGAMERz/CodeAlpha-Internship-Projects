"""
Automated unit tests for Credit Scoring Model.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scorecard import probability_to_score, decision
from features import build_pipeline, engineer_features
from data_loader import load_dataset


def test_scorecard_transformation():
    # Probability 0.0 -> highest score
    high_score = probability_to_score(0.01)
    # Probability 1.0 -> lowest score
    low_score = probability_to_score(0.99)
    assert high_score > low_score
    assert 300 <= low_score <= 850
    assert 300 <= high_score <= 850


def test_decision_tiers():
    threshold = 0.25
    assert decision(0.05, threshold) == "APPROVE"
    assert decision(0.20, threshold) == "MANUAL REVIEW"
    assert decision(0.50, threshold) == "DECLINE"


def test_data_loader_shape():
    X, y = load_dataset()
    assert len(X) > 0
    assert len(y) == len(X)
    assert "LIMIT_BAL" in X.columns
    assert "AGE" in X.columns

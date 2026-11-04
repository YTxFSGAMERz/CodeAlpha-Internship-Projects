"""
Automated unit tests for Speech Emotion Recognition features and pipeline.
"""
import importlib.util
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _import_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

feat_mod = _import_local("speech_features", "features.py")
waveform_to_features = feat_mod.waveform_to_features
augment_waveform = feat_mod.augment_waveform
N_FEATURES = feat_mod.N_FEATURES


def test_waveform_to_features_shape():
    # 2 seconds of 22050 Hz dummy sine wave
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feats = waveform_to_features(y, sr=sr)
    assert feats.ndim == 2
    assert feats.shape[1] == N_FEATURES  # 120 features (40 MFCC + 40 delta + 40 delta-delta)
    assert feats.shape[0] > 0


def test_augment_waveform():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    rng = np.random.default_rng(123)

    aug = augment_waveform(y, sr, rng)
    assert len(aug) > 0
    assert aug.dtype in (np.float32, np.float64)

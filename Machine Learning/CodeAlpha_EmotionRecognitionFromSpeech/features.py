"""
Audio -> features.

Every clip becomes a (frames x 120) matrix: 40 MFCCs + their first and second derivatives
(delta, delta-delta) for each ~23 ms frame. MFCCs summarise the SHAPE of the vocal-tract
spectrum on a perceptual (mel) scale; the deltas capture how it changes - which is where much
of the emotion (rising pitch, trembling, speed) lives.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np

SR = 22050                 # sample rate every clip is converted to
N_MFCC = 40
N_FFT = 2048
HOP = 512                  # 512 / 22050 = 23 ms between frames
MAX_SECONDS = 3.0
MAX_FRAMES = 1 + int(MAX_SECONDS * SR) // HOP      # 130 frames
N_FEATURES = 3 * N_MFCC


def load_audio(path, sr=SR):
    """Load, convert to mono 22.05 kHz, cut leading / trailing silence, cap at 3 seconds."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y[: int(MAX_SECONDS * sr)]


def waveform_to_features(y, sr=SR):
    """waveform -> (frames, 120) float32 matrix, NOT yet padded (frames <= MAX_FRAMES)."""
    if len(y) < sr // 2:                                    # delta needs a few frames
        y = np.pad(y, (0, sr // 2 - len(y)))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak                                        # same loudness for every clip
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    feats = np.vstack([mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]).T
    return feats[:MAX_FRAMES].astype(np.float32)


def augment_waveform(y, sr, rng):
    """One random change to a training clip: noise, pitch shift, time stretch or time shift."""
    choice = int(rng.integers(0, 4))
    if choice == 0:
        return y + rng.uniform(0.002, 0.01) * np.max(np.abs(y)) * rng.standard_normal(len(y))
    if choice == 1:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(rng.uniform(-2, 2)))
    if choice == 2:
        return librosa.effects.time_stretch(y, rate=float(rng.uniform(0.85, 1.15)))
    return np.roll(y, int(rng.uniform(-0.1, 0.1) * sr))


def pad(feats, length=MAX_FRAMES):
    """Stack variable-length (frames, F) matrices into one (N, length, F) array + true lengths."""
    out = np.zeros((len(feats), length, feats[0].shape[1]), dtype=np.float32)
    lengths = np.zeros(len(feats), dtype=np.int32)
    for i, f in enumerate(feats):
        out[i, : len(f)] = f
        lengths[i] = len(f)
    return out, lengths


def extract_files(paths, cache_dir=None, augment_copies=0, seed=0, desc=""):
    """
    Extract features for many files (with an on-disk cache so re-runs are instant).
    With augment_copies > 0 every file additionally produces that many augmented versions.
    Returns (X, lengths, source_index) - source_index[i] = which file row i came from.
    """
    key = hashlib.md5(("|".join(map(str, paths)) + f"|{augment_copies}|{seed}|{SR}|{N_MFCC}|{HOP}").encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"features_{key}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        print(f"  loaded cached features ({desc}) from {cache.name}")
        return z["X"], z["lengths"], z["source"]

    rng = np.random.default_rng(seed)
    feats, source = [], []
    for i, p in enumerate(paths):
        y = load_audio(p)
        feats.append(waveform_to_features(y))
        source.append(i)
        for _ in range(augment_copies):
            feats.append(waveform_to_features(augment_waveform(y, SR, rng)))
            source.append(i)
        if (i + 1) % 200 == 0 or i + 1 == len(paths):
            print(f"  {desc}: {i + 1}/{len(paths)} files processed")
    X, lengths = pad(feats)
    source = np.array(source)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, lengths=lengths, source=source)
    return X, lengths, source


def fit_normalizer(X, lengths):
    """Mean / std of every feature, computed on the real (un-padded) frames of the TRAINING set."""
    mask = np.arange(X.shape[1])[None, :] < lengths[:, None]
    return X[mask].mean(axis=0), X[mask].std(axis=0) + 1e-6


def apply_normalizer(X, lengths, mean, std):
    mask = np.arange(X.shape[1])[None, :] < lengths[:, None]
    Xn = (X - mean) / std
    Xn[~mask] = 0.0                                          # padding stays exactly zero
    return Xn.astype(np.float32)

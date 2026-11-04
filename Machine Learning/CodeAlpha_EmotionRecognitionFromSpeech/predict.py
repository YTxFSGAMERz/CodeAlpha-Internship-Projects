"""
Predict the emotion in a speech recording with the model trained by train.py.

    python predict.py --audio my_voice.wav
    python predict.py --record 4          # record 4 seconds from your microphone first
                                          # (needs:  pip install sounddevice)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np

import features as F

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_bundle():
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras

    model_path = MODEL_DIR / "emotion_model.keras"
    if not model_path.exists():
        raise SystemExit("No trained model found. Run  python train.py --ravdess <folder>  first.")
    norm = np.load(MODEL_DIR / "emotion_norm.npz")
    meta = json.loads((MODEL_DIR / "emotion_meta.json").read_text())
    return keras.models.load_model(model_path), meta, norm


def predict_waveform(model, meta, norm, y):
    """waveform -> probability for every emotion (same steps as in training)."""
    X, lengths = F.pad([F.waveform_to_features(y)])
    X = F.apply_normalizer(X, lengths, norm["mean"], norm["std"])
    return model.predict(X, verbose=0)[0]


def record(seconds, sr=F.SR):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Recording needs the sounddevice package:  pip install sounddevice")
    print(f"Recording for {seconds} s ... speak now!")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    y, _ = librosa.effects.trim(audio[:, 0], top_db=30)
    return y[: int(F.MAX_SECONDS * sr)]


def main():
    ap = argparse.ArgumentParser(description="Predict the emotion in a speech recording.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="path to a .wav (or other audio) file")
    group.add_argument("--record", type=float, metavar="SECONDS", help="record from the microphone")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    model, meta, norm = load_bundle()
    y = F.load_audio(args.audio) if args.audio else record(args.record)
    if len(y) == 0:
        raise SystemExit("The recording is silent.")
    probs = predict_waveform(model, meta, norm, y)

    order = np.argsort(probs)[::-1][: args.top]
    print(f"\nPredicted emotion: {meta['classes'][order[0]].upper()}  ({probs[order[0]]:.1%})\n")
    for i in order:
        print(f"  {meta['classes'][i]:<10} {probs[i]:6.1%}  {'#' * int(probs[i] * 40)}")


if __name__ == "__main__":
    main()

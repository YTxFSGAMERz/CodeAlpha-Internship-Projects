"""
Find the audio files of the three supported emotional-speech datasets and read the emotion
label from the file / folder names. Nothing is downloaded - point the loaders at the folder
you unzipped.

    RAVDESS  Actor_01/03-01-05-01-02-01-12.wav     field 3 = emotion code, last field = actor
    TESS     OAF_angry/OAF_back_angry.wav          last word of the file name = emotion
    EMO-DB   wav/03a01Fa.wav                       6th character = emotion letter, first 2 = speaker

All three are mapped to one shared vocabulary: neutral, calm, happy, sad, angry, fear,
disgust, surprise (+ boredom for EMO-DB).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAVDESS_EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
TESS_EMOTIONS = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
                 "neutral": "neutral", "ps": "surprise", "surprise": "surprise",
                 "surprised": "surprise", "sad": "sad"}
EMODB_EMOTIONS = {"W": "angry", "L": "boredom", "E": "disgust", "A": "fear",
                  "F": "happy", "T": "sad", "N": "neutral"}


def _rows(root, parser, name):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{name} folder not found: {root}")
    rows = []
    for path in sorted(root.rglob("*.wav")):
        parsed = parser(path)
        if parsed:
            rows.append({"path": str(path), "emotion": parsed[0], "speaker": parsed[1], "dataset": name})
    if not rows:
        raise FileNotFoundError(f"No usable .wav files found for {name} under {root}")
    return pd.DataFrame(rows)


def _parse_ravdess(path, speech_only=True):
    parts = path.stem.split("-")
    if len(parts) != 7 or parts[2] not in RAVDESS_EMOTIONS:
        return None
    if speech_only and parts[1] != "01":          # 01 = speech, 02 = song
        return None
    return RAVDESS_EMOTIONS[parts[2]], f"ravdess_{parts[6]}"


def _parse_tess(path):
    emotion = TESS_EMOTIONS.get(path.stem.split("_")[-1].lower())
    if emotion is None:                            # fall back to the folder name, e.g. OAF_Pleasant_surprise
        emotion = TESS_EMOTIONS.get(path.parent.name.split("_")[-1].lower())
    if emotion is None:
        return None
    return emotion, f"tess_{path.stem.split('_')[0].upper()}"


def _parse_emodb(path):
    stem = path.stem
    if len(stem) < 6 or stem[5].upper() not in EMODB_EMOTIONS:
        return None
    return EMODB_EMOTIONS[stem[5].upper()], f"emodb_{stem[:2]}"


def load_ravdess(root):
    return _rows(root, _parse_ravdess, "ravdess")


def load_tess(root):
    return _rows(root, _parse_tess, "tess")


def load_emodb(root):
    return _rows(root, _parse_emodb, "emodb")


def load_datasets(ravdess=None, tess=None, emodb=None, emotions=None):
    """Combine any subset of the three datasets into one table: path, emotion, speaker, dataset."""
    frames = []
    if ravdess:
        frames.append(load_ravdess(ravdess))
    if tess:
        frames.append(load_tess(tess))
    if emodb:
        frames.append(load_emodb(emodb))
    if not frames:
        raise ValueError("Give at least one dataset folder: --ravdess, --tess and/or --emodb")
    df = pd.concat(frames, ignore_index=True)
    if emotions:
        df = df[df["emotion"].isin(emotions)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"None of the files have the emotions {emotions}")
    return df

"""
Speech emotion recognition  (RAVDESS / TESS / EMO-DB)
=====================================================
    python train.py --ravdess data/RAVDESS
    python train.py --tess data/TESS --model cnn
    python train.py --emodb data/emodb --epochs 80
    python train.py --ravdess data/RAVDESS --tess data/TESS --augment 2     # combine + augment

Pipeline
  1. read the emotion label of every .wav from its file name,
  2. split into train / validation / test - by SPEAKER when there are enough speakers, so the
     test set contains voices the model has never heard (the honest way to measure this task),
  3. turn each clip into MFCC + delta + delta-delta features (features.py),
  4. train a CNN, LSTM or CNN-LSTM (model.py) with class weights, early stopping on validation
     accuracy and optional data augmentation of the training clips,
  5. report accuracy, unweighted average recall, per-class precision / recall / F1 and a
     confusion matrix; save the model for predict.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

import features as F
from dataset_loader import load_datasets
from model import build_model

SEED = 42
ROOT = Path(__file__).resolve().parent


def split_indices(df, mode):
    """Return train / validation / test row indices (80 / 20, then 15 % of train for validation)."""
    idx = np.arange(len(df))
    if mode == "auto":
        mode = "speaker" if df["speaker"].nunique() >= 8 else "random"
    if mode == "speaker":
        train_val, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                               .split(idx, groups=df["speaker"]))
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
                      .split(train_val, groups=df["speaker"].iloc[train_val]))
        return train_val[tr], train_val[va], test, mode
    train_val, test = train_test_split(idx, test_size=0.2, stratify=df["emotion"], random_state=SEED)
    tr, va = train_test_split(train_val, test_size=0.15, stratify=df["emotion"].iloc[train_val],
                              random_state=SEED)
    return tr, va, test, mode


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, classes, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(classes)), display_labels=classes,
        normalize="true", values_format=".0%", cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Confusion matrix on unseen speakers (each row sums to 100 %)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    ap.add_argument("--ravdess", help="folder with the RAVDESS audio (Actor_01 ... Actor_24)")
    ap.add_argument("--tess", help="folder with the TESS audio")
    ap.add_argument("--emodb", help="folder with the EMO-DB wav files")
    ap.add_argument("--emotions", nargs="+", help="keep only these emotions, e.g. happy sad angry neutral")
    ap.add_argument("--model", default="cnn_lstm", choices=["cnn", "lstm", "cnn_lstm"])
    ap.add_argument("--split", default="auto", choices=["auto", "speaker", "random"],
                    help="speaker = test on unseen voices; auto picks it when there are >= 8 speakers")
    ap.add_argument("--augment", type=int, default=0, metavar="N",
                    help="add N randomly augmented copies of every training clip (slower)")
    ap.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping may end sooner)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--name", help="name of this run's folder inside outputs/ (default: from the settings)")
    ap.add_argument("--no-cache", action="store_true", help="do not cache extracted features in ./cache")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    (ROOT / "models").mkdir(exist_ok=True)
    cache = None if args.no_cache else ROOT / "cache"

    df = load_datasets(args.ravdess, args.tess, args.emodb, args.emotions)
    run_name = args.name or "_".join(sorted(df["dataset"].unique()) + [args.model]
                                     + ([f"aug{args.augment}"] if args.augment else []))
    out = ROOT / "outputs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(df["emotion"].unique())
    y_all = df["emotion"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    print(f"{len(df)} clips, {df['speaker'].nunique()} speakers, {len(classes)} emotions: {classes}")
    print(df["emotion"].value_counts().to_string(), "\n")

    tr, va, te, mode = split_indices(df, args.split)
    print(f"Split by {mode}: {len(tr)} train / {len(va)} validation / {len(te)} test clips "
          f"({df['speaker'].iloc[te].nunique()} test speakers)\n")

    paths = df["path"].to_numpy()
    print("Extracting MFCC features ...")
    X_tr, L_tr, src = F.extract_files(paths[tr], cache, args.augment, SEED, "train")
    y_tr = y_all[tr][src]                                   # labels follow the augmented copies
    X_va, L_va, _ = F.extract_files(paths[va], cache, desc="validation")
    X_te, L_te, _ = F.extract_files(paths[te], cache, desc="test")
    y_va, y_te = y_all[va], y_all[te]

    mean, std = F.fit_normalizer(X_tr, L_tr)                # statistics from TRAINING data only
    X_tr, X_va, X_te = (F.apply_normalizer(X, L, mean, std)
                        for X, L in [(X_tr, L_tr), (X_va, L_va), (X_te, L_te)])
    print(f"Training tensor: {X_tr.shape}  (clips, frames, features)\n")

    model = build_model(args.model, X_tr.shape[1:], len(classes))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    # Early stopping watches validation ACCURACY. (Validation loss is a poor signal on small speech
    # datasets: it is lowest in the first epochs, before the model has learned anything useful.)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=25,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs,
                        batch_size=args.batch_size, class_weight=dict(enumerate(map(float, weights))),
                        callbacks=callbacks, verbose=2)

    probs = model.predict(X_te, batch_size=64, verbose=0)
    y_pred = probs.argmax(axis=1)
    accuracy = float((y_pred == y_te).mean())
    uar = float(balanced_accuracy_score(y_te, y_pred))          # unweighted average recall
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    print(f"\nTEST accuracy {accuracy:.3f} | unweighted average recall {uar:.3f} | macro F1 {macro_f1:.3f}\n")
    report = classification_report(y_te, y_pred, labels=range(len(classes)), target_names=classes,
                                   digits=3, zero_division=0)
    print(report)
    (out / "classification_report.txt").write_text(report)
    plot_history(history, out / "training_curves.png")
    plot_confusion(y_te, y_pred, classes, out / "confusion_matrix.png")

    model.save(ROOT / "models" / "emotion_model.keras")
    np.savez(ROOT / "models" / "emotion_norm.npz", mean=mean, std=std)
    meta = {"classes": classes, "model": args.model, "split": mode, "sample_rate": F.SR,
            "n_mfcc": F.N_MFCC, "hop_length": F.HOP, "max_frames": F.MAX_FRAMES,
            "datasets": sorted(df["dataset"].unique().tolist()),
            "test_accuracy": round(accuracy, 4), "test_uar": round(uar, 4),
            "test_macro_f1": round(macro_f1, 4), "train_clips": int(len(tr)),
            "augmented_copies": args.augment, "epochs_trained": len(history.history["loss"])}
    (ROOT / "models" / "emotion_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models/emotion_model.keras and plots in outputs/{run_name}/")


if __name__ == "__main__":
    main()

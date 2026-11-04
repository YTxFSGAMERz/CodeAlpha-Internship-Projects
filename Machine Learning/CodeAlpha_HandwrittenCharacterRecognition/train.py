"""
Handwritten character recognition with a Convolutional Neural Network
=====================================================================
    python train.py --dataset mnist                    # handwritten digits 0-9
    python train.py --dataset emnist-balanced          # digits + upper/lower-case letters (47 classes)
    python train.py --dataset emnist-letters           # letters A-Z (26 classes)
    python train.py --dataset digits-demo --epochs 5   # tiny OFFLINE smoke test (not real MNIST)

What it does
  1. loads the dataset and splits off a validation set,
  2. builds a CNN (with random rotation / shift / zoom augmentation while training),
  3. trains with early stopping and a learning-rate schedule,
  4. evaluates on the untouched test set: accuracy, per-class precision / recall / F1,
     confusion matrix, and a gallery of the mistakes,
  5. saves the model (models/<dataset>_cnn.keras) plus plots in outputs/<dataset>/.
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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from data_loader import SEED, load_dataset
from model import build_cnn

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- plots
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
    many = len(classes) > 15
    size = 12 if many else 7
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(classes)), display_labels=classes,
        normalize="true" if many else None, include_values=not many,
        cmap="Blues", colorbar=False, ax=ax, xticks_rotation="vertical" if many else "horizontal")
    ax.set_title("Confusion matrix (rows = true character, columns = predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gallery(x, titles, path, heading, cols=5, rows=5):
    n = min(len(x), cols * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t in zip(np.ravel(axes), x[:n], titles[:n]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(t, fontsize=8)
    fig.suptitle(heading)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train a CNN for handwritten character recognition.")
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "emnist-digits", "emnist-letters", "emnist-balanced",
                             "emnist-byclass", "digits-demo", "custom"])
    ap.add_argument("--npz", help="with --dataset custom: .npz file with x_train, y_train, x_test, y_test")
    ap.add_argument("--epochs", type=int, default=20, help="maximum epochs (early stopping may end sooner)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, help="use only this many training images (quick experiments)")
    ap.add_argument("--no-augment", action="store_true", help="turn off random rotation / shift / zoom")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    out = ROOT / "outputs" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    data = load_dataset(args.dataset, limit=args.limit, npz_path=args.npz)
    classes = data["class_names"]
    print(f"Dataset {args.dataset}: {len(data['x_train']):,} train / {len(data['x_val']):,} validation / "
          f"{len(data['x_test']):,} test images, {len(classes)} classes")

    model = build_cnn(len(classes), augment=not args.no_augment)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(data["x_train"], data["y_train"],
                        validation_data=(data["x_val"], data["y_val"]),
                        batch_size=args.batch_size, shuffle=True,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)

    # ---- evaluation on data the model has never seen
    test_loss, test_acc = model.evaluate(data["x_test"], data["y_test"],
                                         batch_size=args.batch_size, verbose=0)
    probs = model.predict(data["x_test"], batch_size=512, verbose=0)
    y_true, y_pred = data["y_test"], probs.argmax(axis=1)
    print(f"\nTEST ACCURACY: {test_acc:.4f}   (loss {test_loss:.4f})\n")
    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    print(report)
    (out / "classification_report.txt").write_text(report)

    plot_history(history, out / "training_curves.png")
    plot_confusion(y_true, y_pred, classes, out / "confusion_matrix.png")

    wrong = np.where(y_pred != y_true)[0]
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])]      # most confident mistakes first
    if len(wrong):
        plot_gallery(data["x_test"][wrong],
                     [f"true {classes[y_true[i]]} / pred {classes[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})"
                      for i in wrong],
                     out / "misclassified_examples.png", f"Mistakes ({len(wrong)} of {len(y_true)} test images)")
    pick = np.random.default_rng(SEED).choice(len(y_true), 25, replace=False)
    plot_gallery(data["x_test"][pick],
                 [f"{classes[y_true[i]]} -> {classes[y_pred[i]]}" for i in pick],
                 out / "sample_predictions.png", "Random test images (true -> predicted)")

    model.save(ROOT / "models" / f"{args.dataset}_cnn.keras")
    meta = {"dataset": args.dataset, "class_names": classes, "preprocess": data["preprocess"],
            "test_accuracy": round(float(test_acc), 4), "test_loss": round(float(test_loss), 4),
            "epochs_trained": len(history.history["loss"]), "train_images": int(len(data["x_train"]))}
    (ROOT / "models" / f"{args.dataset}_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models/{args.dataset}_cnn.keras and plots in outputs/{args.dataset}/")


if __name__ == "__main__":
    main()

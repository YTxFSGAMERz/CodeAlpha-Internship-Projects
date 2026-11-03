"""Step 2 - train the LSTM on the sequences made by preprocess.py.

    python train.py                       # up to 40 epochs, stops early when it stops improving
    python train.py --epochs 10 --units 128
    python train.py --resume              # carry on training the saved model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib
import numpy as np

matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt            # noqa: E402
import keras                               # noqa: E402

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from model import build_model                            # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=192, help="size of each LSTM layer")
    parser.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="share of pieces kept for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="continue from the model saved in models/")
    args = parser.parse_args()

    data_file = PROCESSED_DIR / "sequences.npz"
    if not data_file.exists():
        raise SystemExit("Run `python preprocess.py` first.")
    data = np.load(data_file)
    vocab = json.loads((PROCESSED_DIR / "vocab.json").read_text())
    n_chords, n_durations = len(vocab["chords"]), len(vocab["durations"])

    # Validate on whole pieces the network never trains on (windows of one piece overlap).
    rng = np.random.default_rng(args.seed)
    pieces = np.unique(data["piece_id"])
    val_pieces = rng.choice(pieces, size=max(1, int(len(pieces) * args.val_fraction)), replace=False)
    is_val = np.isin(data["piece_id"], val_pieces)

    def split(mask):
        return ({"chord_in": data["chord_in"][mask], "dur_in": data["dur_in"][mask]},
                {"chord_out": data["chord_out"][mask], "dur_out": data["dur_out"][mask]})

    x_train, y_train = split(~is_val)
    x_val, y_val = split(is_val)
    print(f"{len(y_train['chord_out']):,} training sequences, {len(y_val['chord_out']):,} validation sequences, "
          f"{n_chords} chords, {n_durations} waits")

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_file = MODEL_DIR / "music_lstm.keras"
    shutil.copy(PROCESSED_DIR / "vocab.json", MODEL_DIR / "vocab.json")

    keras.utils.set_random_seed(args.seed)
    if args.resume and model_file.exists():
        print(f"Resuming from {model_file}")
        model = keras.models.load_model(model_file)
    else:
        model = build_model(n_chords, n_durations, units=args.units)
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            # the best model so far is saved after every epoch, so an interrupted run loses nothing
            keras.callbacks.ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-4),
        ],
        verbose=2,
    )
    model.save(model_file)

    # A few real windows that generate.py can use as the starting point of a new piece.
    pick = rng.choice(len(data["chord_in"]), size=min(200, len(data["chord_in"])), replace=False)
    np.savez_compressed(MODEL_DIR / "seeds.npz", chord_in=data["chord_in"][pick], dur_in=data["dur_in"][pick])

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(history.history["loss"], label="training")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].plot(history.history["chord_out_accuracy"], label="training")
    axes[1].plot(history.history["val_chord_out_accuracy"], label="validation")
    axes[1].set(title="Next-chord accuracy", xlabel="epoch")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=120)

    best = int(np.argmin(history.history["val_loss"]))
    print(f"\nBest epoch {best + 1}: validation loss {history.history['val_loss'][best]:.3f}, "
          f"next-chord accuracy {history.history['val_chord_out_accuracy'][best]:.1%}")
    print(f"Model saved to {MODEL_DIR / 'music_lstm.keras'}")


if __name__ == "__main__":
    main()

"""
Recognise a handwritten character from an image file.

    python predict.py my_digit.png                            # model trained on MNIST
    python predict.py letter.png --dataset emnist-letters
    python predict.py photo.jpg --dataset emnist-balanced --top 5 --save-processed seen_by_model.png

Tips: write ONE character, big and thick (marker or a thick brush in Paint), dark ink on plain paper.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess import to_mnist_format

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_model_and_meta(dataset: str):
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras

    model_path = MODEL_DIR / f"{dataset}_cnn.keras"
    meta_path = MODEL_DIR / f"{dataset}_meta.json"
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path}. Run:  python train.py --dataset {dataset}")
    return keras.models.load_model(model_path), json.loads(meta_path.read_text())


def predict_images(model, meta, images28, top=3):
    """images28: list of (28, 28) uint8 arrays -> for each, the `top` (character, probability) pairs."""
    probs = model.predict(np.stack(images28)[..., None], verbose=0)
    results = []
    for p in probs:
        order = np.argsort(p)[::-1][:top]
        results.append([(meta["class_names"][i], float(p[i])) for i in order])
    return results


def main():
    ap = argparse.ArgumentParser(description="Recognise a handwritten character.")
    ap.add_argument("image", help="path to an image containing ONE handwritten character")
    ap.add_argument("--dataset", default="mnist", help="which trained model to use (default: mnist)")
    ap.add_argument("--top", type=int, default=3, help="how many guesses to show")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="auto = detect dark-ink-on-light-paper and flip it")
    ap.add_argument("--save-processed", help="save the 28x28 image the model actually sees")
    args = ap.parse_args()

    model, meta = load_model_and_meta(args.dataset)
    invert = {"auto": None, "yes": True, "no": False}[args.invert]
    img28 = to_mnist_format(args.image, invert=invert, **meta["preprocess"])
    if args.save_processed:
        Image.fromarray(img28).resize((224, 224), Image.NEAREST).save(args.save_processed)

    guesses = predict_images(model, meta, [img28], top=args.top)[0]
    print(f"\nPrediction: '{guesses[0][0]}'  ({guesses[0][1]:.1%} confident)")
    for char, p in guesses:
        print(f"  {char!r:>5}  {p:6.1%}  {'#' * int(p * 40)}")


if __name__ == "__main__":
    main()

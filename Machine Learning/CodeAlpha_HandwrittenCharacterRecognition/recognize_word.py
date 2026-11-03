"""
BONUS - read a whole word (or line) by cutting it into characters and classifying each one.

    python recognize_word.py word.png --dataset emnist-letters
    python recognize_word.py number.png --dataset mnist --save word_result.png

Works when the characters are separated by small gaps (printed, or written neatly apart).
Joined-up (cursive) handwriting needs a sequence model - a CRNN trained with CTC loss on a
word-level dataset such as IAM; see the README ("Extending to words and sentences").
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw

from predict import load_model_and_meta, predict_images
from preprocess import load_gray, needs_inversion, otsu_threshold, to_mnist_format


def find_character_boxes(ink: np.ndarray, min_width=2, merge_gap=1):
    """ink: boolean array (True = ink). Returns [(x0, x1, y0, y1), ...] left to right.
    Characters are the runs of columns that contain ink; runs split by <= merge_gap empty columns
    are merged (so a letter that has a tiny break is not cut in two)."""
    has_ink = ink.any(axis=0)
    runs, start = [], None
    for x, flag in enumerate(has_ink):
        if flag and start is None:
            start = x
        elif not flag and start is not None:
            runs.append([start, x - 1])
            start = None
    if start is not None:
        runs.append([start, len(has_ink) - 1])

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    boxes = []
    for x0, x1 in merged:
        if x1 - x0 + 1 < min_width:
            continue                                   # specks of noise
        rows = np.where(ink[:, x0:x1 + 1].any(axis=1))[0]
        boxes.append((x0, x1, int(rows.min()), int(rows.max())))
    return boxes


def space_cutoff(gaps, height):
    """Gap size (pixels) above which a gap is a space between words, or None if there are none.
    Splits the gaps into a 'small' and a 'large' group (1-D 2-means); it only counts as a
    word space when the large group is clearly bigger and at least half a character high."""
    if not gaps:
        return None
    if len(gaps) == 1:
        return height                                  # one gap: compare with the character height
    g = np.asarray(gaps, dtype=float)
    lo, hi = g.min(), g.max()
    if hi < 1.8 * max(lo, 1.0):
        return None                                    # all gaps alike -> a single word
    for _ in range(20):
        mid = (lo + hi) / 2
        small, large = g[g <= mid], g[g > mid]
        if len(small) == 0 or len(large) == 0:
            break
        lo, hi = small.mean(), large.mean()
    return max((lo + hi) / 2, 0.5 * height) if hi > 1.8 * lo else None


def recognize_word(path, dataset="emnist-letters", merge_gap=1, save=None):
    model, meta = load_model_and_meta(dataset)
    gray = load_gray(path)
    if needs_inversion(gray):
        gray = 255.0 - gray
    gray = (gray - gray.min()) / max(gray.max() - gray.min(), 1) * 255.0
    ink = gray > otsu_threshold(gray.astype(np.uint8))

    boxes = find_character_boxes(ink, merge_gap=merge_gap)
    if not boxes:
        raise SystemExit("No characters found in the image.")
    crops = [to_mnist_format(gray[y0:y1 + 1, x0:x1 + 1], invert=False, **meta["preprocess"])
             for x0, x1, y0, y1 in boxes]
    guesses = predict_images(model, meta, crops, top=1)

    # A space is a gap clearly wider than the usual gap between letters of one word.
    gaps = [boxes[i][0] - boxes[i - 1][1] - 1 for i in range(1, len(boxes))]
    height = float(np.median([y1 - y0 + 1 for _, _, y0, y1 in boxes]))
    cutoff = space_cutoff(gaps, height)
    text = ""
    for i, g in enumerate(guesses):
        if i and cutoff is not None and gaps[i - 1] > cutoff:
            text += " "
        text += g[0][0]

    if save:
        canvas = Image.fromarray(255 - gray.astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for (x0, x1, y0, y1), g in zip(boxes, guesses):
            draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(220, 40, 40))
            draw.text((x0, max(0, y0 - 12)), g[0][0], fill=(220, 40, 40))
        canvas.save(save)
    return text, [(g[0][0], g[0][1]) for g in guesses]


def main():
    ap = argparse.ArgumentParser(description="Recognise a handwritten word by segmenting characters.")
    ap.add_argument("image")
    ap.add_argument("--dataset", default="emnist-letters", help="trained model to use (letters or digits)")
    ap.add_argument("--merge-gap", type=int, default=1, help="merge character pieces closer than this many pixels")
    ap.add_argument("--save", help="save an image with the detected boxes and predicted letters")
    args = ap.parse_args()
    text, per_char = recognize_word(args.image, args.dataset, args.merge_gap, args.save)
    print(f"\nRecognised text: {text!r}")
    print("  per character:", "  ".join(f"{c}({p:.0%})" for c, p in per_char))


if __name__ == "__main__":
    main()

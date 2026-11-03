"""
Turn a photo / scan / drawing of handwriting into the 28x28 white-on-black format of MNIST/EMNIST.

This file only needs numpy, scipy and Pillow (no TensorFlow), so it is easy to test on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift as nd_shift

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def load_gray(source) -> np.ndarray:
    """Path, PIL image or array  ->  2-D float array (0-255)."""
    if isinstance(source, (str, Path)):
        source = Image.open(source)
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("L"), dtype=np.float32)
    arr = np.asarray(source, dtype=np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def needs_inversion(gray: np.ndarray) -> bool:
    """MNIST ink is white on black. A pen on paper is dark on light, so flip it.
    The median pixel is the background, because ink covers only a small part of the picture."""
    return float(np.median(gray)) > 127


def otsu_threshold(gray_u8: np.ndarray) -> int:
    """Automatic threshold that best separates ink from background."""
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    s0 = np.cumsum(hist * levels)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return int(np.argmax(w0 * w1 * (m0 - m1) ** 2))


def to_mnist_format(source, invert=None, ink_size=20, center_by_mass=True, threshold=0.15) -> np.ndarray:
    """
    ONE character -> (28, 28) uint8 array, white ink on black.

    1. make the ink white on black (auto-detected unless `invert` is given)
    2. stretch the contrast and cut away the empty border (bounding box of the ink)
    3. shrink so the longer side is `ink_size` pixels, keeping the aspect ratio
    4. centre it on a 28x28 canvas (by centre of mass for MNIST-style models)
    """
    gray = load_gray(source)
    if invert is None:
        invert = needs_inversion(gray)
    if invert:
        gray = 255.0 - gray
    gray = gray - gray.min()
    if gray.max() <= 0:
        raise ValueError("The image is blank - no handwriting found.")
    gray = gray / gray.max() * 255.0

    ys, xs = np.where(gray > threshold * 255)
    if len(ys) == 0:
        raise ValueError("No handwriting found in the image.")
    glyph = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = glyph.shape
    scale = ink_size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    glyph = np.asarray(Image.fromarray(glyph.astype(np.uint8)).resize((new_w, new_h), _LANCZOS),
                       dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = glyph
    if center_by_mass and canvas.sum() > 0:
        cy, cx = center_of_mass(canvas)
        canvas = nd_shift(canvas, (13.5 - cy, 13.5 - cx), order=1, mode="constant", cval=0.0)
    return np.clip(canvas, 0, 255).astype(np.uint8)

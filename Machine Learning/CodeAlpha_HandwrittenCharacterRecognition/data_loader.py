"""
Dataset loading for handwritten character recognition.

    mnist            digits 0-9                          60,000 train / 10,000 test
    emnist-digits    EMNIST digits                       240,000 / 40,000
    emnist-letters   letters A-Z (upper/lower merged)    124,800 / 20,800
    emnist-balanced  digits + letters, 47 classes        112,800 / 18,800
    emnist-byclass   digits + upper + lower, 62 classes  697,932 / 116,323
    digits-demo      scikit-learn's tiny 8x8 digits resized to 28x28 - an OFFLINE smoke test only
    custom           your own .npz file (x_train, y_train, x_test, y_test)

Images are returned as uint8 arrays of shape (N, 28, 28, 1): white ink on black, upright.
The model rescales to [0, 1] itself, so training and prediction use the same raw pixel values.
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

DIGITS = list("0123456789")
UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_NAMES = {
    "mnist": DIGITS,
    "emnist-digits": DIGITS,
    "emnist-letters": UPPER,
    "emnist-balanced": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"),   # 47 classes
    "emnist-byclass": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    "digits-demo": DIGITS,
}

# How predict.py should prepare a photo so it resembles the training images.
# MNIST digits fill a 20x20 box and are centred by centre of mass; EMNIST glyphs fill more of the
# 28x28 frame and are centred by bounding box (24 px is a reasonable choice, not an official figure).
PREPROCESS = {
    "mnist": {"ink_size": 20, "center_by_mass": True},
    "digits-demo": {"ink_size": 20, "center_by_mass": True},
    "emnist-digits": {"ink_size": 24, "center_by_mass": False},
    "emnist-letters": {"ink_size": 24, "center_by_mass": False},
    "emnist-balanced": {"ink_size": 24, "center_by_mass": False},
    "emnist-byclass": {"ink_size": 24, "center_by_mass": False},
    "custom": {"ink_size": 20, "center_by_mass": True},
}


def _load_mnist():
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
    return x_tr, y_tr, x_te, y_te


def _load_emnist(subset):
    try:
        from emnist import extract_test_samples, extract_training_samples
    except ImportError:
        raise SystemExit("EMNIST needs the 'emnist' package:  pip install emnist\n"
                         "(it downloads the ~535 MB EMNIST archive once and caches it)")
    x_tr, y_tr = extract_training_samples(subset)         # the package already fixes the orientation
    x_te, y_te = extract_test_samples(subset)
    if subset == "letters":                               # raw EMNIST letter labels are 1..26
        y_tr, y_te = y_tr - 1, y_te - 1
    return x_tr, y_tr, x_te, y_te


def _load_digits_demo():
    from scipy.ndimage import zoom
    from sklearn.datasets import load_digits
    d = load_digits()
    imgs = np.stack([np.clip(zoom(im / 16.0 * 255.0, 28 / 8, order=1), 0, 255) for im in d.images])
    imgs = imgs.astype(np.uint8)
    return train_test_split(imgs, d.target, test_size=0.25, stratify=d.target, random_state=SEED)


def _load_custom(path):
    with np.load(path, allow_pickle=False) as f:
        return f["x_train"], f["y_train"], f["x_test"], f["y_test"]


def load_dataset(name="mnist", val_fraction=0.1, limit=None, npz_path=None):
    """Return a dict with train / validation / test splits and the class names."""
    if name == "mnist":
        x_tr, y_tr, x_te, y_te = _load_mnist()
    elif name.startswith("emnist-"):
        if name not in CLASS_NAMES:
            raise ValueError(f"Unknown EMNIST subset '{name}'. Options: {[k for k in CLASS_NAMES if k.startswith('emnist')]}")
        x_tr, y_tr, x_te, y_te = _load_emnist(name.split("-", 1)[1])
    elif name == "digits-demo":
        x_tr, x_te, y_tr, y_te = _load_digits_demo()
    elif name == "custom":
        if not npz_path:
            raise ValueError("--dataset custom needs --npz path/to/file.npz")
        x_tr, y_tr, x_te, y_te = _load_custom(npz_path)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(CLASS_NAMES) + ['custom']}")

    x_tr, x_te = x_tr.astype(np.uint8), x_te.astype(np.uint8)
    y_tr, y_te = y_tr.astype(np.int64), y_te.astype(np.int64)

    if limit and limit < len(x_tr):                       # quick experiments on a subset
        keep = np.random.default_rng(SEED).permutation(len(x_tr))[:limit]
        x_tr, y_tr = x_tr[keep], y_tr[keep]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=val_fraction, stratify=y_tr, random_state=SEED)

    classes = CLASS_NAMES.get(name) or [str(i) for i in range(int(max(y_tr.max(), y_te.max())) + 1)]
    return {
        "x_train": x_tr[..., None], "y_train": y_tr,
        "x_val": x_val[..., None], "y_val": y_val,
        "x_test": x_te[..., None], "y_test": y_te,
        "class_names": classes,
        "preprocess": PREPROCESS.get(name, PREPROCESS["custom"]),
    }

"""
Automated unit tests for Handwritten Character Recognition preprocessing and segmentation.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess import needs_inversion, otsu_threshold, to_mnist_format
from recognize_word import find_character_boxes


def test_needs_inversion():
    # Light background (e.g. white page 255) needs inversion
    white_page = np.full((100, 100), 250, dtype=np.float32)
    white_page[40:60, 40:60] = 0  # black ink
    assert needs_inversion(white_page) is True

    # Dark background (e.g. black page 0) does not need inversion
    black_board = np.full((100, 100), 10, dtype=np.float32)
    black_board[40:60, 40:60] = 250 # white chalk
    assert needs_inversion(black_board) is False


def test_otsu_threshold():
    # Bimodal distribution with realistic ink and background clusters
    rng = np.random.default_rng(42)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:50, :] = rng.integers(180, 220, (50, 100), dtype=np.uint8)
    image[50:, :] = rng.integers(10, 40, (50, 100), dtype=np.uint8)
    thresh = otsu_threshold(image)
    assert 35 <= thresh <= 180


def test_to_mnist_format_dimensions():
    # Synthetic drawing of a stroke
    canvas = np.zeros((64, 64), dtype=np.float32)
    canvas[20:44, 30:34] = 255.0  # vertical line (ink)
    mnist_char = to_mnist_format(canvas, invert=False)
    assert mnist_char.shape == (28, 28)
    assert mnist_char.dtype == np.uint8
    assert mnist_char.max() > 0


def test_character_box_segmentation():
    # Two distinct characters separated by gap
    ink = np.zeros((50, 100), dtype=bool)
    # Character 1: cols 10..20
    ink[10:40, 10:21] = True
    # Character 2: cols 40..50
    ink[10:40, 40:51] = True
    boxes = find_character_boxes(ink, min_width=2, merge_gap=2)
    assert len(boxes) == 2
    assert boxes[0][0] == 10 and boxes[0][1] == 20
    assert boxes[1][0] == 40 and boxes[1][1] == 50

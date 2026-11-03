"""Tests for the SORT tracker. Run with: pytest"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sort_tracker import SortTracker, associate, iou_matrix  # noqa: E402


def box(x, y, w=40, h=80, conf=0.9, cls=0):
    """One detection row: x1, y1, x2, y2, confidence, class."""
    return [x, y, x + w, y + h, conf, cls]


def ids_by_row(tracks, y_split=90):
    """Return {'top': id, 'bottom': id} using each box's vertical position."""
    found = {}
    for track in tracks:
        centre_y = (track.bbox[1] + track.bbox[3]) / 2
        found["top" if centre_y < y_split else "bottom"] = track.track_id
    return found


# --------------------------------------------------------------------- geometry
def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0
    assert iou_matrix(a, np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)


def test_iou_with_no_boxes_has_the_right_shape():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_association_pairs_the_overlapping_boxes():
    detections = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    predictions = np.array([[101, 101, 111, 111], [1, 1, 11, 11]], float)
    matches, unmatched_detections, unmatched_tracks = associate(detections, predictions, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched_detections == [] and unmatched_tracks == []


# ------------------------------------------------------------------- identities
def test_ids_stay_the_same_for_two_moving_objects():
    rng = np.random.default_rng(0)
    tracker = SortTracker()
    seen = {"top": set(), "bottom": set()}
    for frame in range(40):
        jitter = rng.normal(0, 1.0, size=4)
        detections = np.array([box(10 + 5 * frame + jitter[0], 20 + jitter[1]),
                               box(300 - 4 * frame + jitter[2], 100 + jitter[3])], float)
        for name, track_id in ids_by_row(tracker.update(detections)).items():
            seen[name].add(track_id)
    assert len(seen["top"]) == 1 and len(seen["bottom"]) == 1
    assert seen["top"] != seen["bottom"]


def test_a_new_object_gets_a_new_id():
    tracker = SortTracker()
    ids = set()
    for frame in range(10):
        ids |= {t.track_id for t in tracker.update(np.array([box(10 + 3 * frame, 20)], float))}
    assert len(ids) == 1
    for frame in range(10, 20):
        detections = np.array([box(10 + 3 * frame, 20), box(250, 120)], float)
        ids |= {t.track_id for t in tracker.update(detections)}
    assert len(ids) == 2


def test_object_keeps_its_id_after_a_short_disappearance():
    tracker = SortTracker(max_age=30)
    before = set()
    for frame in range(10):
        before |= {t.track_id for t in tracker.update(np.array([box(10 + 5 * frame, 20)], float))}
    for _ in range(5):                                      # hidden for 5 frames
        assert tracker.update(np.empty((0, 6))) == []
    # it re-appears where its constant speed says it should be
    after = {t.track_id for t in tracker.update(np.array([box(10 + 5 * 15, 20)], float))}
    after |= {t.track_id for t in tracker.update(np.array([box(10 + 5 * 16, 20)], float))}
    after |= {t.track_id for t in tracker.update(np.array([box(10 + 5 * 17, 20)], float))}
    assert len(before) == 1 and after == before


def test_object_gets_a_new_id_after_it_was_gone_longer_than_max_age():
    tracker = SortTracker(max_age=10)
    first = set()
    for frame in range(10):
        first |= {t.track_id for t in tracker.update(np.array([box(50, 50)], float))}
    for _ in range(25):
        tracker.update(np.empty((0, 6)))
    second = set()
    for _ in range(5):
        second |= {t.track_id for t in tracker.update(np.array([box(50, 50)], float))}
    assert first and second and first.isdisjoint(second)


def test_a_one_frame_false_alarm_is_never_shown():
    tracker = SortTracker(min_hits=3)
    for _ in range(6):
        tracker.update(np.empty((0, 6)))
    assert tracker.update(np.array([box(30, 30)], float)) == []
    assert tracker.update(np.empty((0, 6))) == []


def test_a_new_object_is_shown_after_min_hits_detections():
    tracker = SortTracker(min_hits=3)
    for _ in range(6):
        tracker.update(np.empty((0, 6)))
    shown = [len(tracker.update(np.array([box(30, 30)], float))) for _ in range(4)]
    assert shown == [0, 0, 1, 1]


def test_class_label_is_decided_by_vote():
    tracker = SortTracker()
    last = []
    for frame, cls in enumerate([2, 2, 2, 7, 2, 2]):
        last = tracker.update(np.array([box(10 + 2 * frame, 20, cls=cls)], float))
    assert last[0].class_id == 2


def test_empty_input_does_not_crash():
    tracker = SortTracker()
    assert tracker.update(np.empty((0, 6))) == []
    assert tracker.update([]) == []


def test_a_confirmed_object_does_not_flicker_after_one_missed_frame():
    tracker = SortTracker(min_hits=3)
    for frame in range(8):
        tracker.update(np.array([box(10 + 4 * frame, 20)], float))
    assert tracker.update(np.empty((0, 6))) == []                     # the detector missed it once
    again = tracker.update(np.array([box(10 + 4 * 9, 20)], float))    # ...and it is shown again straight away
    assert len(again) == 1

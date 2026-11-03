"""Tests for the video pipeline and the detector adapters. No webcam, GPU or YOLO weights needed."""

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_track import run  # noqa: E402
from detector import YoloDetector, resolve_classes  # noqa: E402
from sort_tracker import SortTracker, TrackedObject  # noqa: E402

NAMES = {0: "person", 2: "car"}


def make_video(path: Path, frames: int = 30, size=(320, 240)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        x = 10 + 6 * i
        cv2.rectangle(frame, (x, 80), (x + 50, 160), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class FakeDetector:
    """Pretends to be YOLO: it knows where the white rectangle is in every frame."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        x = 10 + 6 * self.calls
        self.calls += 1
        return np.array([[x, 80, x + 50, 160, 0.9, 0]], dtype=np.float32)


def test_pipeline_tracks_saves_and_summarises(tmp_path):
    video, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(video)

    summary = run(FakeDetector(), SortTracker(), str(video), NAMES, show=False, save=output, trails=True)

    assert summary == {"frames": 30, "unique_objects": 1, "by_class": {"person": 1}}
    capture = cv2.VideoCapture(str(output))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 30
    last = frames[-1].astype(int)
    assert np.abs(last[..., 0] - last[..., 2]).max() > 60, "a coloured box and label should be drawn"


def test_max_frames_stops_early(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video)
    assert run(FakeDetector(), SortTracker(), str(video), NAMES, show=False, max_frames=7)["frames"] == 7


def test_missing_video_gives_a_clear_error(tmp_path):
    with pytest.raises(SystemExit):
        run(FakeDetector(), SortTracker(), str(tmp_path / "nope.mp4"), NAMES, show=False)


def test_resolve_classes_accepts_names_and_numbers():
    assert resolve_classes(["car", "0"], NAMES) == [2, 0]
    assert resolve_classes(None, NAMES) is None
    with pytest.raises(SystemExit):
        resolve_classes(["unicorn"], NAMES)


# ---- the YOLO wrapper, with a stand-in for the ultralytics package ----
class _Boxes:
    def __init__(self, n):
        self._n = n
        self.xyxy = np.array([[10, 20, 30, 40], [50, 60, 90, 120]], float)[:n]
        self.conf = np.array([0.9, 0.5])[:n]
        self.cls = np.array([0.0, 2.0])[:n]

    def __len__(self):
        return self._n


class _FakeYOLO:
    boxes_to_return = 2

    def __init__(self, weights):
        self.names = dict(NAMES)
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [types.SimpleNamespace(boxes=_Boxes(self.boxes_to_return))]


def test_yolo_wrapper_returns_boxes_in_one_array(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    detector = YoloDetector("fake.pt", conf=0.4, imgsz=320, classes=["car"])
    result = detector(np.zeros((10, 10, 3), np.uint8))

    assert result.shape == (2, 6)
    assert result[1].tolist() == pytest.approx([50, 60, 90, 120, 0.5, 2])
    call = detector.model.calls[0]
    assert call["conf"] == 0.4 and call["imgsz"] == 320 and call["classes"] == [2] and call["verbose"] is False


def test_yolo_wrapper_handles_frames_without_detections(monkeypatch):
    class Empty(_FakeYOLO):
        boxes_to_return = 0

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Empty))
    assert YoloDetector("fake.pt")(np.zeros((10, 10, 3), np.uint8)).shape == (0, 6)


# ---- the Deep SORT adapter, with a stand-in for deep-sort-realtime ----
def test_deepsort_adapter_converts_boxes_both_ways(monkeypatch):
    class FakeTrack:
        track_id, det_class, det_conf, time_since_update = "7", 2, 0.8, 0

        def is_confirmed(self):
            return True

        def to_ltrb(self):
            return (10, 20, 50, 100)

    class FakeDeepSort:
        def __init__(self, **kwargs):
            pass

        def update_tracks(self, raw, frame=None):
            FakeDeepSort.received = raw
            return [FakeTrack()]

    module = types.ModuleType("deep_sort_realtime.deepsort_tracker")
    module.DeepSort = FakeDeepSort
    monkeypatch.setitem(sys.modules, "deep_sort_realtime", types.ModuleType("deep_sort_realtime"))
    monkeypatch.setitem(sys.modules, "deep_sort_realtime.deepsort_tracker", module)

    from deepsort_tracker import DeepSortTracker

    shown = DeepSortTracker().update(np.array([[10, 20, 50, 100, 0.8, 2]], np.float32), frame=None)
    assert FakeDeepSort.received == [([10.0, 20.0, 40.0, 80.0], pytest.approx(0.8), 2)]   # left, top, width, height
    assert shown == [TrackedObject(7, (10.0, 20.0, 50.0, 100.0), 2, pytest.approx(0.8))]

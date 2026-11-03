"""SORT (Simple Online and Realtime Tracking), written from scratch with NumPy.

A detector such as YOLO only says "there are boxes in this frame". It has no idea
which box in frame 10 is the same person as a box in frame 9. SORT adds that memory:

1. PREDICT  every tracked object owns a Kalman filter that guesses where its box
            will be in the new frame, from the box position and speed so far.
2. MATCH    new detections are paired with the predicted boxes by overlap (IoU).
            The Hungarian algorithm picks the best pairing over all boxes at once.
3. UPDATE   a matched track is corrected with its detection; a detection nobody
            claimed starts a new track (a new ID); a track that finds no detection
            for `max_age` frames is deleted.

Reference: Bewley et al., "Simple Online and Realtime Tracking", 2016.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackedObject:
    """One box with a stable ID, as returned by every tracker in this project."""

    track_id: int
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    class_id: int
    confidence: float


# --------------------------------------------------------------- geometry
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection-over-union of every box in `boxes_a` with every box in `boxes_b`."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    width = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    height = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = width * height
    union = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
             + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter)
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def associate(detections: np.ndarray, predictions: np.ndarray, iou_threshold: float):
    """Pair detections with predicted boxes. Returns (matches, unmatched_dets, unmatched_tracks)."""
    if len(predictions) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(predictions)))

    overlap = iou_matrix(detections, predictions)
    rows, cols = linear_sum_assignment(-overlap)          # maximise the total overlap
    matches = [(d, t) for d, t in zip(rows, cols) if overlap[d, t] >= iou_threshold]
    matched_d = {d for d, _ in matches}
    matched_t = {t for _, t in matches}
    return (
        matches,
        [d for d in range(len(detections)) if d not in matched_d],
        [t for t in range(len(predictions)) if t not in matched_t],
    )


# ----------------------------------------------------------- one object
def _box_to_state(box) -> np.ndarray:
    """[x1, y1, x2, y2] -> [centre x, centre y, area, aspect ratio] as a column."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return np.array([[x1 + width / 2.0], [y1 + height / 2.0], [width * height], [width / max(height, 1e-6)]])


class _Track:
    """A Kalman filter following one object.

    State: [centre x, centre y, area, aspect ratio, speed x, speed y, area change].
    The aspect ratio is assumed constant; the other three change at a constant speed.
    """

    _F = np.eye(7)                  # how the state moves from one frame to the next
    _F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
    _H = np.eye(4, 7)               # we only measure the first four numbers

    def __init__(self, box, track_id: int, class_id: int, confidence: float) -> None:
        self.id = track_id
        self.x = np.zeros((7, 1))
        self.x[:4] = _box_to_state(box)

        self.P = np.eye(7)                     # uncertainty: unknown speeds are very uncertain
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)                     # process noise: speeds change only slowly
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)                     # measurement noise: size is noisier than position
        self.R[2:, 2:] *= 10.0

        self.hits = 1                          # detections matched so far (the first one counts)
        self.hit_streak = 1                    # ... in a row
        self.confirmed = False                 # becomes True once, after `min_hits` detections in a row
        self.time_since_update = 0
        self.confidence = confidence
        self._votes: dict[int, float] = defaultdict(float)
        self._votes[class_id] += confidence

    @property
    def class_id(self) -> int:
        """Most-voted class, so a label that flickers between frames settles down."""
        return max(self._votes, key=self._votes.get)

    @property
    def box(self) -> np.ndarray:
        cx, cy, area, ratio = self.x[:4, 0]
        width = np.sqrt(max(area * ratio, 1e-6))
        height = max(area, 1e-6) / width
        return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])

    def predict(self) -> np.ndarray:
        if self.x[6, 0] + self.x[2, 0] <= 0:   # never let the area become negative
            self.x[6, 0] = 0.0
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        if self.time_since_update > 0:
            self.hit_streak = 0                # a gap breaks the streak
        self.time_since_update += 1
        return self.box

    def update(self, box, class_id: int, confidence: float) -> None:
        innovation = _box_to_state(box) - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self.R
        gain = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + gain @ innovation
        self.P = (np.eye(7) - gain @ self._H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self._votes[class_id] += confidence


# ----------------------------------------------------------- the tracker
class SortTracker:
    """Give every detected object an ID that stays the same from frame to frame.

    max_age        frames a track survives without a matching detection (hidden objects)
    min_hits       detections in a row needed before a new track is shown (hides one-frame false alarms)
    iou_threshold  minimum overlap between a prediction and a detection to count as the same object
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[_Track] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, detections, frame=None) -> list[TrackedObject]:
        """Feed one frame of detections: an (N, 6) array of x1, y1, x2, y2, confidence, class_id.

        `frame` is unused here; it is accepted so SORT and Deep SORT share one interface.
        """
        self.frame_count += 1
        detections = np.asarray(detections, dtype=float).reshape(-1, 6)

        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)
        usable = ~np.isnan(predicted).any(axis=1)
        self.tracks = [t for t, ok in zip(self.tracks, usable) if ok]
        predicted = predicted[usable]

        matches, unmatched, _ = associate(detections[:, :4], predicted, self.iou_threshold)
        for d, t in matches:
            self.tracks[t].update(detections[d, :4], int(detections[d, 5]), float(detections[d, 4]))
        for d in unmatched:
            self.tracks.append(_Track(detections[d, :4], self._next_id, int(detections[d, 5]), float(detections[d, 4])))
            self._next_id += 1

        for track in self.tracks:
            if track.hit_streak >= self.min_hits:
                track.confirmed = True         # once confirmed, a track is not hidden again after a missed frame

        shown = [
            TrackedObject(t.id, tuple(float(v) for v in t.box), t.class_id, t.confidence)
            for t in self.tracks
            if t.time_since_update == 0 and (t.confirmed or self.frame_count <= self.min_hits)
        ]
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return shown

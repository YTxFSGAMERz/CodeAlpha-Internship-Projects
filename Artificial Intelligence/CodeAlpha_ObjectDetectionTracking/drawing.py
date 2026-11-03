"""Everything drawn on top of the video: boxes, labels, trails and the info panel."""

from __future__ import annotations

import colorsys
from collections import deque

import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """A bright BGR colour that is always the same for the same ID."""
    hue = (track_id * 0.61803398875) % 1.0           # golden-ratio steps keep neighbours distinct
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame: np.ndarray, tracks, names: dict[int, str]) -> None:
    """Draw a box and an "#ID class confidence" label for each tracked object."""
    height, width = frame.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        color = color_for_id(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {names.get(track.class_id, track.class_id)} {track.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        top = y1 - text_h - baseline - 4
        if top < 0:                                   # box touches the top edge: put the label inside
            top = y1
        left = min(max(x1, 0), max(width - text_w - 6, 0))
        cv2.rectangle(frame, (left, top), (left + text_w + 6, top + text_h + baseline + 4), color, -1)
        luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        ink = (0, 0, 0) if luminance > 150 else (255, 255, 255)
        cv2.putText(frame, label, (left + 3, top + text_h + 1), FONT, 0.5, ink, 1, cv2.LINE_AA)


class Trails:
    """Remembers where each ID has been and draws it as a fading line under the object's feet."""

    def __init__(self, length: int = 30, forget_after: int = 60) -> None:
        self.length = length
        self.forget_after = forget_after
        self._points: dict[int, deque] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def update(self, tracks) -> None:
        self._frame += 1
        for track in tracks:
            x1, _, x2, y2 = track.bbox
            self._points.setdefault(track.track_id, deque(maxlen=self.length)).append((int((x1 + x2) / 2), int(y2)))
            self._last_seen[track.track_id] = self._frame
        for track_id in [i for i, seen in self._last_seen.items() if self._frame - seen > self.forget_after]:
            del self._points[track_id], self._last_seen[track_id]

    def draw(self, frame: np.ndarray) -> None:
        for track_id, points in self._points.items():
            color = color_for_id(track_id)
            for start, end, step in zip(points, list(points)[1:], range(1, len(points))):
                thickness = max(1, int(3 * step / len(points)))
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_info(frame: np.ndarray, fps: float, visible: int, total: int) -> None:
    """Small panel in the top-left corner."""
    lines = [f"FPS {fps:4.1f}", f"Tracking {visible}", f"IDs so far {total}"]
    for i, text in enumerate(lines):
        position = (10, 24 + 22 * i)
        cv2.putText(frame, text, position, FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)       # outline
        cv2.putText(frame, text, position, FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    print("=" * 60)
    print("CodeAlpha - Object Detection & Tracking Visualizer (drawing.py)")
    print("=" * 60)
    print("Testing HUD telemetry and annotation drawing on a synthetic frame...")
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_info(canvas, fps=29.8, visible=3, total=7)
    print(f"Canvas created and annotated: {canvas.shape[1]}x{canvas.shape[0]} px, 3 channels.")
    print("\nNote: This is an annotation module. To run live tracking, execute:")
    print("    python detect_track.py --source 0")
    print("    python detect_track.py --source your_video.mp4 --trails")

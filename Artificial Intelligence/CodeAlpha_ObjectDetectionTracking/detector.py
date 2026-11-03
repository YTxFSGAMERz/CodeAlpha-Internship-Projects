"""YOLO object detector (Ultralytics) with a tiny interface: frame in, boxes out."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    for _dir in sys.path:
        _libs = os.path.join(_dir, "numpy.libs")
        if os.path.isdir(_libs):
            try:
                os.add_dll_directory(_libs)
            except Exception:
                pass
            os.environ["PATH"] = _libs + os.pathsep + os.environ.get("PATH", "")

import numpy as np


def resolve_classes(requested, names: dict[int, str]) -> list[int] | None:
    """Turn ["person", "car"] or ["0", "2"] into class ids. None means "detect everything"."""
    if not requested:
        return None
    by_name = {name.lower(): class_id for class_id, name in names.items()}
    ids = []
    for item in requested:
        text = str(item).strip().lower()
        if text.isdigit() and int(text) in names:
            ids.append(int(text))
        elif text in by_name:
            ids.append(by_name[text])
        else:
            examples = ", ".join(list(by_name)[:8])
            raise SystemExit(f"Unknown class '{item}'. Some valid names: {examples}, ...")
    return ids


def _to_numpy(values) -> np.ndarray:
    """Ultralytics returns PyTorch tensors; accept plain arrays as well."""
    return values.cpu().numpy() if hasattr(values, "cpu") else np.asarray(values)


class YoloDetector:
    """Wraps a YOLO model. Calling it with a BGR frame returns an (N, 6) array:
    x1, y1, x2, y2, confidence, class_id."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640,
                 classes=None, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise SystemExit("Ultralytics is not installed. Run:  pip install -r requirements.txt") from error

        local_weights = Path(__file__).parent / weights
        if not Path(weights).is_file() and local_weights.is_file():
            weights = str(local_weights)

        self.model = YOLO(weights)          # downloads the weights the first time if not found
        self.names: dict[int, str] = dict(self.model.names)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_ids = resolve_classes(classes, self.names)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, classes=self.class_ids,
            device=self.device, verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack([
            _to_numpy(boxes.xyxy),
            _to_numpy(boxes.conf),
            _to_numpy(boxes.cls),
        ]).astype(np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("CodeAlpha - YOLO Object Detector Module (detector.py)")
    print("=" * 60)
    print("Initializing YOLO detector...")
    try:
        det = YoloDetector(conf=0.35)
        print(f"Detector ready. Total categories: {len(det.names)}")
        print(f"Sample classes: {', '.join([det.names[i] for i in range(min(8, len(det.names)))])}...")
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = det(sample_frame)
        print(f"Test inference on 640x480 frame: {len(dets)} detections found (black test image).")
    except Exception as e:
        print(f"Detector check: {e}")
    print("\nNote: This is a support module. To run the full detection and tracking program, execute:")
    print("    python detect_track.py --source 0")
    print("    python detect_track.py --source your_video.mp4")

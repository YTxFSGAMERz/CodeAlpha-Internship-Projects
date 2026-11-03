"""YOLO object detector (Ultralytics) with a tiny interface: frame in, boxes out."""

from __future__ import annotations

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

        self.model = YOLO(weights)          # downloads the weights the first time
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

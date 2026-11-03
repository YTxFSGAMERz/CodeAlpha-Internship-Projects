"""Deep SORT, through the `deep-sort-realtime` package (optional).

Deep SORT is SORT plus a small neural network that looks at what each box *looks
like*. Matching by appearance as well as position keeps the same ID when people
cross paths or an object is hidden for a moment, but it is slower than plain SORT.

Install the extra package first:   pip install deep-sort-realtime
"""

from __future__ import annotations

from sort_tracker import TrackedObject


class DeepSortTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.3) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as error:
            raise SystemExit("Deep SORT needs one more package:  pip install deep-sort-realtime") from error
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",     # the appearance network; runs on the CPU if there is no GPU
            bgr=True,                 # OpenCV frames are BGR
        )

    def update(self, detections, frame=None) -> list[TrackedObject]:
        """detections: (N, 6) array of x1, y1, x2, y2, confidence, class_id; frame: the BGR image."""
        raw = [
            ([x1, y1, x2 - x1, y2 - y1], conf, int(class_id))     # Deep SORT wants left, top, width, height
            for x1, y1, x2, y2, conf, class_id in detections
        ]
        tracks = self._tracker.update_tracks(raw, frame=frame)

        shown = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue                                           # still unproven, or hidden this frame
            left, top, right, bottom = track.to_ltrb()
            shown.append(TrackedObject(
                track_id=int(track.track_id),
                bbox=(float(left), float(top), float(right), float(bottom)),
                class_id=int(track.det_class) if track.det_class is not None else -1,
                confidence=float(track.det_conf or 0.0),
            ))
        return shown

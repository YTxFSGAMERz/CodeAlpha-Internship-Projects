"""Real-time object detection and tracking.

    python detect_track.py                              # webcam, YOLO + SORT
    python detect_track.py --source street.mp4          # a video file
    python detect_track.py --source street.mp4 --save out.mp4 --no-show
    python detect_track.py --classes person car --trails
    python detect_track.py --tracker deepsort           # needs: pip install deep-sort-realtime

Press q or Esc in the video window to quit.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from deepsort_tracker import DeepSortTracker
from detector import YoloDetector
from drawing import Trails, draw_info, draw_tracks
from sort_tracker import SortTracker

WINDOW = "Object detection and tracking (q to quit)"


def open_source(source: str) -> cv2.VideoCapture:
    """A number means a webcam, anything else is a video file or stream URL."""
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        if source.isdigit():
            raise SystemExit(
                f"Could not open webcam #{source}. If no camera is available on this machine, "
                f"pass a video file instead:  python detect_track.py --source your_video.mp4"
            )
        raise SystemExit(f"Could not open the video file or stream: {source}")
    return capture


def run(detector, tracker, source: str, names: dict[int, str], show: bool = True,
        save: Path | None = None, trails: bool = False, max_frames: int | None = None) -> dict:
    """Read frames, detect, track, draw, then show and/or save them. Returns a summary."""
    capture = open_source(source)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    trail_memory = Trails() if trails else None
    class_of_id: dict[int, int] = {}                 # every ID ever shown -> its latest class
    frames, fps = 0, 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()

            detections = detector(frame)                      # 1. where are the objects?
            tracked = tracker.update(detections, frame)       # 2. which one is which?

            for track in tracked:
                class_of_id[track.track_id] = track.class_id
            if trail_memory:
                trail_memory.update(tracked)
                trail_memory.draw(frame)
            draw_tracks(frame, tracked, names)

            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed   # smoothed
            draw_info(frame, fps, len(tracked), len(class_of_id))

            if save is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
                writer.write(frame)
            if show:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    by_class = Counter(names.get(class_id, str(class_id)) for class_id in class_of_id.values())
    return {"frames": frames, "unique_objects": len(class_of_id), "by_class": dict(by_class)}


def create_tracker(name: str, args):
    if name == "deepsort":
        return DeepSortTracker(max_age=args.max_age, n_init=args.min_hits)
    return SortTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam number (default 0) or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights, e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="minimum detection confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="size the frame is resized to for detection")
    parser.add_argument("--classes", nargs="+", help="only these classes, by name or number (e.g. person car)")
    parser.add_argument("--device", default=None, help="cpu, 0 (first GPU), mps ... (default: automatic)")
    parser.add_argument("--tracker", choices=["sort", "deepsort"], default="sort")
    parser.add_argument("--max-age", type=int, default=30, help="frames to remember an object that disappeared")
    parser.add_argument("--min-hits", type=int, default=3, help="detections in a row before a new object gets an ID")
    parser.add_argument("--iou", type=float, default=0.3, help="SORT: minimum overlap to match a box to a track")
    parser.add_argument("--trails", action="store_true", help="draw the path each object has taken")
    parser.add_argument("--save", type=Path, help="write the annotated video to this .mp4 file")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    args = parser.parse_args()

    detector = YoloDetector(args.model, args.conf, args.imgsz, args.classes, args.device)
    tracker = create_tracker(args.tracker, args)
    summary = run(detector, tracker, args.source, detector.names, show=not args.no_show,
                  save=args.save, trails=args.trails, max_frames=args.max_frames)

    print(f"\nProcessed {summary['frames']} frames, tracked {summary['unique_objects']} different objects")
    for name, count in sorted(summary["by_class"].items(), key=lambda item: -item[1]):
        print(f"  {name}: {count}")
    if args.save:
        print(f"Annotated video saved to {args.save}")


if __name__ == "__main__":
    main()

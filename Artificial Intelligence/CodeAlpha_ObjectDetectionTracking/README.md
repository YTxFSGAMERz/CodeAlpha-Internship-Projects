# Object Detection and Tracking

**CodeAlpha Artificial Intelligence Internship - Task 4**

Detects objects in a webcam stream or a video file with a pre-trained **YOLO** model, follows each object from frame to frame with the **SORT** tracking algorithm, and shows the result live: a box around every object with its **class name, confidence and tracking ID**.

```
 webcam / video file ──► YOLO (detect) ──► SORT (track) ──► boxes + labels + IDs on screen
        OpenCV          boxes per frame     same ID per object      OpenCV window / saved video
```

## Features

- Real-time input from a webcam or a video file (OpenCV)
- Object detection with a pre-trained YOLO model from Ultralytics (80 everyday classes such as person, car, dog, bottle)
- Object tracking with **SORT**, implemented from scratch in `sort_tracker.py` (Kalman filter + Hungarian matching)
- Optional **Deep SORT** tracker (`--tracker deepsort`) that also looks at appearance
- Label on every box: `#ID class confidence`, a different colour for every ID
- Optional movement trails (`--trails`), FPS counter, count of tracked objects
- Save the annotated video (`--save`), run without a window (`--no-show`), track only some classes (`--classes person car`)

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt` installs PyTorch through Ultralytics, so the first install is large (a few hundred MB to a few GB, depending on your system). The YOLO weights (`yolov8n.pt`, about 6 MB) are downloaded automatically the first time you run the program.

```bash
python detect_track.py                                   # webcam 0
python detect_track.py --source my_video.mp4             # video file
python detect_track.py --source my_video.mp4 --save result.mp4 --no-show
python detect_track.py --classes person car --trails     # only people and cars, with paths
python detect_track.py --model yolov8s.pt --conf 0.5     # bigger model, stricter threshold
python detect_track.py --tracker deepsort                # needs: pip install deep-sort-realtime
```

Press **q** or **Esc** in the video window to quit. When the program ends it prints how many different objects it tracked.

| Option | Meaning | Default |
| --- | --- | --- |
| `--source` | webcam number or video path | `0` |
| `--model` | YOLO weights (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`, ...) | `yolov8n.pt` |
| `--conf` | minimum detection confidence | `0.35` |
| `--classes` | only track these classes (names or numbers) | all |
| `--tracker` | `sort` or `deepsort` | `sort` |
| `--max-age` | frames an object may disappear and still keep its ID | `30` |
| `--min-hits` | detections in a row before a new object gets an ID | `3` |
| `--iou` | SORT: minimum overlap to match a detection to a track | `0.3` |
| `--trails`, `--save`, `--no-show`, `--max-frames`, `--imgsz`, `--device` | see `python detect_track.py --help` | |

## How it works

**1. Detection (YOLO).** For every frame the model returns boxes with a class and a confidence. It has no memory: a person in frame 10 and the same person in frame 11 are just two unrelated boxes.

**2. Tracking (SORT).** `sort_tracker.py` gives the boxes a memory:

1. **Predict** - each tracked object has a Kalman filter that guesses where its box will be in this frame, based on its position and speed so far.
2. **Match** - the new detections are compared with the predicted boxes using IoU (how much two boxes overlap). The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the best pairing for all boxes together.
3. **Update** - matched tracks are corrected with their detection and keep their ID. A detection that matches nothing starts a new track with a new ID. A track that is not matched for `--max-age` frames is deleted. A new track is only shown after `--min-hits` detections in a row, which hides one-frame false alarms.

The class shown on a box is decided by a vote over all frames, so a label does not flicker between "car" and "truck".

**3. Display (OpenCV).** `drawing.py` draws the boxes, labels, trails and the information panel, and `detect_track.py` shows and/or saves the frames.

### SORT or Deep SORT?

| | SORT (default) | Deep SORT |
| --- | --- | --- |
| Matches by | position and speed | position, speed **and appearance** |
| Speed | very fast | slower (runs a second neural network) |
| Extra install | none | `pip install deep-sort-realtime` |
| Best for | clear scenes, real-time on a CPU | crowded scenes, objects that cross paths or hide briefly |

## Project structure

```
CodeAlpha_ObjectDetectionTracking/
├── detect_track.py       # command line program and main video loop
├── detector.py           # YOLO wrapper: frame in, boxes out
├── sort_tracker.py       # SORT tracker (Kalman filter + Hungarian algorithm)
├── deepsort_tracker.py   # optional Deep SORT adapter
├── drawing.py            # boxes, labels, trails, info panel
├── tests/
└── requirements.txt
```

## Tests

```bash
pytest
```

The tests cover the tracker (IDs stay stable, new objects get new IDs, short disappearances, false alarms, label voting) and the whole video pipeline using a stand-in detector, so they run without a webcam, a GPU or any model download.

## Tips and limitations

- **Speed**: use `yolov8n.pt` (the smallest model), lower `--imgsz` (for example `480`), or run on a GPU with `--device 0`.
- **Accuracy**: use a larger model such as `yolov8s.pt`, or raise `--conf` to remove weak detections.
- SORT only knows positions. If two people cross paths, or someone leaves the frame and comes back after more than `--max-age` frames, they can get a new ID. Deep SORT is better at this.
- YOLO can only detect the 80 classes it was trained on (the COCO dataset).

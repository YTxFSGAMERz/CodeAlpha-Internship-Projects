# Emotion Recognition from Speech

**CodeAlpha Machine Learning Internship - Task 2**
Author: Farhan Shaikh

Recognise human emotions (angry, happy, sad, ...) from speech audio using **MFCC features** and deep learning
(**CNN, LSTM or CNN-LSTM**) on the RAVDESS, TESS and EMO-DB datasets.

## Datasets

The datasets are not included - download one (or more) and point the script at the unzipped folder.

| Dataset | Clips | Speakers | Emotions | How the label is read |
|---|---|---|---|---|
| **RAVDESS** (speech) | 1,440 | 24 actors | neutral, calm, happy, sad, angry, fear, disgust, surprise | file `03-01-05-01-02-01-12.wav`: 3rd number = emotion, last = actor |
| **TESS** | 2,800 | 2 actresses | angry, disgust, fear, happy, neutral, sad, surprise | last word of the file name (`OAF_back_angry.wav`) |
| **EMO-DB** | 535 | 10 actors | angry, boredom, disgust, fear, happy, neutral, sad | 6th letter of `03a01Fa.wav` (F = happy) |

`dataset_loader.py` maps all three to one vocabulary, so you can train on a single dataset or combine them:
`--ravdess <folder> --tess <folder> --emodb <folder>`; `--emotions happy sad angry neutral` keeps only some emotions.

## Approach

1. **Audio -> features** (`features.py`): load at 22.05 kHz mono, cut leading/trailing silence, cap at 3 s, then compute
   **40 MFCCs** for every 23 ms frame plus their first and second derivatives (**delta, delta-delta**) - a
   130 x 120 matrix per clip. MFCCs describe the shape of the vocal-tract spectrum on the mel (perceptual) scale;
   the deltas capture how it changes over time (rising pitch, trembling, speaking speed), which is where much of
   the emotion lives. Features are standardised with statistics from the *training* clips only.
2. **Models** (`model.py`, choose with `--model`):
   - `cnn` - three Conv1D blocks (BatchNorm, ReLU, MaxPool, Dropout), then average + max pooling over time
   - `lstm` - two stacked bidirectional LSTMs read the MFCC sequence directly
   - `cnn_lstm` (default) - Conv1D blocks find local patterns, a bidirectional LSTM models how they evolve
3. **Speaker-independent split** - when there are at least 8 speakers the clips are split **by speaker**
   (about 60 % / 20 % / 20 % of the speakers for train / validation / test), so the test set contains voices the model has
   never heard. This is the honest way to measure the task; a random split lets the model recognise the *speaker*
   and inflates accuracy. TESS has only two speakers, so it falls back to a random stratified split
   (`--split speaker|random` overrides).
4. **Training** - Adam, class weights for imbalanced emotions, learning-rate reduction on plateau, and **early stopping
   on validation accuracy**. (Validation *loss* is a bad stopping signal on small speech datasets: it is lowest in the
   first epochs, before anything useful is learned - an early version of this script stopped there and scored at chance.)
5. **Optional augmentation** (`--augment N`) - N extra copies of every *training* clip with random noise, pitch shift,
   time stretch or time shift.
6. **Evaluation** - accuracy, **unweighted average recall (UAR)** (the mean of the per-emotion recalls - the standard
   fair score when classes are imbalanced), macro F1, per-emotion precision / recall / F1, confusion matrix.

## Results

### EMO-DB - measured in my sandbox

EMO-DB was the one dataset I could run end to end: 535 clips, 7 emotions, split by speaker into 317 training clips (6 speakers),
104 validation clips (2 speakers) and **114 test clips from 2 speakers the model never heard**. Chance level is about 14 %.

| Run | Accuracy | UAR | Macro F1 |
|---|---|---|---|
| `cnn_lstm`, no augmentation | 0.596 | 0.570 | 0.547 |
| `cnn_lstm`, `--augment 2` (951 training clips) | 0.605 | 0.587 | 0.523 |

The two runs are not meaningfully different - 114 clips from two speakers is a small, noisy test set (a few clips
move the numbers by several points). Anger is recognised best (F1 0.85-0.92) and sad is rarely missed; the quiet,
low-energy emotions (neutral, boredom) are confused with each other and the rarest class, disgust (5 test clips), is poor -
a typical pattern for speech emotion recognition. Plots and per-emotion reports are in `outputs/emodb_cnn_lstm/` and
`outputs/emodb_cnn_lstm_aug2/`.

### RAVDESS / TESS - fill in after your run

RAVDESS has roughly four times as many clips as EMO-DB, so expect it to train better. Run it and put your numbers here:

| Dataset | Model | Split | Accuracy | UAR | Macro F1 |
|---|---|---|---|---|---|
| RAVDESS | cnn_lstm | by speaker | | | |
| TESS | cnn_lstm | random (2 speakers) | | | |

Every run prints these numbers and saves them to `outputs/<run name>/metrics.json`.

## Run it

```bash
pip install -r requirements.txt

# 1) 1-minute smoke test with synthetic tones in RAVDESS file-naming format (NOT real speech - accuracy is meaningless)
python tools/make_demo_data.py
python train.py --ravdess demo_data/RAVDESS --epochs 15

# 2) real data
python train.py --ravdess path/to/RAVDESS                       # speaker-independent
python train.py --ravdess path/to/RAVDESS --augment 2           # slower, often helps
python train.py --tess path/to/TESS --model cnn
python train.py --emodb path/to/emodb/wav --epochs 100
python train.py --ravdess path/to/RAVDESS --tess path/to/TESS --emotions happy sad angry neutral fear disgust surprise
```

The first run extracts features (a minute or two for RAVDESS) and caches them in `cache/`; later runs are fast.
Training saves `models/emotion_model.keras`, the feature normaliser and a metadata file, plus plots and a classification
report in `outputs/<run name>/`.

### Predict the emotion of a recording

```bash
python predict.py --audio my_voice.wav
python predict.py --record 4        # records 4 s from your microphone first (pip install sounddevice; not tested in my sandbox)
```

## Project structure

```
dataset_loader.py   find the .wav files of RAVDESS / TESS / EMO-DB and read the labels
features.py         audio -> MFCC + delta + delta-delta, augmentation, normalisation, feature cache
model.py            CNN, LSTM and CNN-LSTM networks
train.py            speaker-independent split, training, evaluation, plots, saving
predict.py          emotion of a file or a live recording
tools/make_demo_data.py   synthetic RAVDESS-style mini dataset for a quick smoke test
outputs/            results of my EMO-DB runs
```

## Limitations

- Acted emotions in studio recordings: real-world speech, noise and spontaneous emotion are much harder.
- A model trained on one language/dataset transfers poorly to others (RAVDESS is North American English, EMO-DB is German).
- Small datasets and few speakers make results vary a lot between splits - judge by UAR on unseen speakers.

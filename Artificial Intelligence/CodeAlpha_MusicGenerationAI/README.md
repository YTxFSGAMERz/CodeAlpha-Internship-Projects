# Music Generation with AI

**CodeAlpha Artificial Intelligence Internship - Task 3**

A deep-learning model that learns from MIDI music and composes new music. The network is an **LSTM** (a kind of RNN) built with TensorFlow/Keras, the music is read and written with **music21**, and the result is saved as a **MIDI file** and as a **WAV audio file**.

```
MIDI files ──► preprocess.py ──► train.py ──► generate.py ──► new_piece.mid + new_piece.wav
 (music21)     notes → numbers    LSTM learns    model composes
                                  the patterns   note by note
```

Open `outputs/sample_generated.mid` or `outputs/sample_generated.wav` to hear a piece that this project generated.

## Try it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

python generate.py --wav      # compose with the model that is already in models/
```

The `models/` folder contains a small model trained on Bach chorales, so `generate.py` works straight away. To train your own:

```bash
python preprocess.py          # 1. MIDI -> training sequences
python train.py               # 2. train the LSTM (about 8 minutes on one CPU core in my run)
python generate.py --wav      # 3. compose
```

Useful options for `generate.py`:

| Option | Meaning |
| --- | --- |
| `--length 400` | how many events (chords/notes) to compose, default 200 |
| `--temperature 1.2` | higher = more surprising, lower = safer (default 0.9) |
| `--top-k 12` | each step only chooses among the k most likely chords |
| `--bpm 90` | playback speed |
| `--count 3` | create 3 different pieces |
| `--seed-midi song.mid` | start from the opening of a MIDI file you provide |
| `--wav` | also save an audio file |
| `--seed 7` | repeatable results |

You can also turn any MIDI file into audio with `python midi_to_wav.py song.mid`.

## Using your own MIDI data

Put `.mid` files in `data/midi/` (classical, jazz, film music, anything) and run `preprocess.py` and `train.py` again. If the folder is empty, the project uses the Bach chorales that come with `music21`. More pieces of one style give better results.

## How it works

**1. Music becomes events** (`preprocess.py`). music21 reads every MIDI file. Each piece becomes a list of events, and an event is "play these notes now, then wait this long":

```
((60, 64, 67), 1.0)    play C + E + G together, next event comes one quarter note later
```

- Pieces are first transposed to C major / A minor, so the network learns musical patterns instead of 24 different keys.
- Times are rounded to sixteenth notes, a chord keeps at most 4 notes, and very rare chords are replaced by their top note, which keeps the vocabulary small (1,077 chords for the Bach data).

**2. Events become training examples.** Chords and waits are numbered, and each piece is cut into overlapping windows of 32 events. For every position in the window the target is the *next* event.

**3. The network** (`model.py`) has two inputs and two outputs because an event has two parts:

```
chord number ─► Embedding ─┐
                           ├─► LSTM ─► LSTM ─┬─► softmax ─► probability of every possible next chord
wait number  ─► Embedding ─┘                 └─► softmax ─► probability of every possible next wait
```

The embeddings turn numbers into learnable vectors, the two LSTM layers remember what came before, and dropout reduces overfitting. The loss is a weighted sum of both prediction errors.

**4. Composing** (`generate.py`). The model gets a 32-event starting phrase (a real phrase from the training data, or from your own MIDI file) and predicts probabilities for the next chord and wait. One is picked at random, added to the phrase, and the loop repeats. *Temperature* controls how adventurous the choice is and *top-k* stops very unlikely chords from being picked. The starting phrase is not part of the output.

**5. Saving** (`music_utils.py`). The events are written to a MIDI file with music21. For the WAV file, the notes are rendered with a small synthesizer written in NumPy (a few overtones with a fading envelope), so no sound fonts or extra software are needed.

## Results of the included model

| | |
| --- | --- |
| Data | 369 Bach chorales: 30,630 events, 1,077 different chords |
| Training / validation | 8,442 / 1,065 sequences (validation pieces are never trained on) |
| Network | 2 LSTM layers of 192 units, 779,205 parameters |
| Training | batch 32, about 30-40 s per epoch on one CPU core, stopped early after 10 epochs |
| Best epoch (5) | next-chord accuracy 14.2% (1,077 possible chords), next-wait accuracy 77.6% |

How the generated music compares with the real music it learned from (6 generated pieces of 200 events, measured by `numpy` code, not by ear):

| | Real chorales | Generated |
| --- | --- | --- |
| Notes inside the C-major scale | 93.9% | 95.8% |
| Notes per event | 2.47 | 2.28 |
| Same chord twice in a row | 0.8% | 1.4% |
| 4-chord sequences copied from the training data | - | 2.4% |

So the model stays in key, uses chords and rhythms like the originals and mostly produces new combinations instead of copying. It is still a small model trained on a small dataset: the learning curves in `outputs/training_curves.png` show overfitting after epoch 5 (training loss keeps falling while validation loss rises), so do not expect a finished composition. Every run gives a different piece.

## Ideas to make it better

- Train on more MIDI files of one style (hundreds of pieces rather than a few hundred short chorales).
- More dropout or a smaller network to reduce the overfitting visible in the curves.
- Keep note durations and velocities, or use several instruments (this project plays everything as piano).
- Try a GAN or a Transformer instead of an LSTM.

## Project structure

```
CodeAlpha_MusicGenerationAI/
├── preprocess.py     # MIDI -> events -> training sequences
├── model.py          # the LSTM network
├── train.py          # training (use --resume to continue an interrupted run)
├── generate.py       # compose new music
├── music_utils.py    # events -> MIDI file, events -> WAV audio
├── midi_to_wav.py    # turn any MIDI file into audio
├── config.py         # settings (sequence length, chord limits, folders)
├── data/midi/        # put your MIDI files here
├── models/           # trained model, vocabulary, starting phrases
├── outputs/          # sample piece and training curves
└── tests/
```

## Tests

```bash
pytest
```

They check reading music, transposition, encoding, the shape of the network output, sampling, and writing MIDI and WAV files.

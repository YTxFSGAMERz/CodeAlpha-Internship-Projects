"""Settings shared by every script in the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIDI_DIR = ROOT / "data" / "midi"          # drop your own .mid files here
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# --- how music is turned into numbers -------------------------------------
GRID = 0.25            # smallest time step in quarter notes (a sixteenth note)
MAX_DELTA = 4.0        # longest wait between two events (a whole bar of 4/4)
MAX_POLYPHONY = 4      # at most this many notes are kept in one chord
MIN_CHORD_COUNT = 4    # chords seen fewer times than this are simplified

# --- how sequences are cut for training -----------------------------------
SEQ_LEN = 32           # the model reads this many events before predicting the next
STRIDE = 2             # start a new training sequence every STRIDE events

# Every possible wait between two events: 0.25, 0.5, ... 4.0 quarter notes.
DURATIONS = [GRID * step for step in range(1, int(MAX_DELTA / GRID) + 1)]

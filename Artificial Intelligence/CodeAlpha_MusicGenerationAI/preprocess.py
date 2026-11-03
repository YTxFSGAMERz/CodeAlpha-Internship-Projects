"""Step 1 - turn MIDI music into sequences a neural network can learn from.

Every piece becomes a list of *events*. An event says "play these notes now,
then wait this long before the next event":

    ((60, 64, 67), 1.0)   ->  play C4+E4+G4, next event is one quarter note later

Pieces are transposed to C major / A minor first, so the model learns the
patterns of music instead of 24 different keys. Chords and waits get integer ids
and are cut into overlapping windows of SEQ_LEN events.

    python preprocess.py                 # your files in data/midi, else Bach chorales
    python preprocess.py --source bach   # force the Bach chorales that ship with music21
    python preprocess.py --max-files 50  # quick test run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from music21 import chord, converter, corpus, freezeThaw, note

# Prevent music21 cache permission errors on locked Windows temp files
_orig_freezer_write = freezeThaw.StreamFreezer.write


def _safe_freezer_write(self, *args, **kwargs):
    try:
        return _orig_freezer_write(self, *args, **kwargs)
    except (PermissionError, OSError):
        return None


freezeThaw.StreamFreezer.write = _safe_freezer_write

from config import (
    DURATIONS, GRID, MAX_DELTA, MAX_POLYPHONY, MIDI_DIR, MIN_CHORD_COUNT,
    PROCESSED_DIR, SEQ_LEN, STRIDE,
)


# ------------------------------------------------------------ reading music
def find_midi_files(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.suffix.lower() in {".mid", ".midi"}]
    return sorted(files)


def key_shift(score) -> int:
    """Semitones to move the piece so that it ends up in C major or A minor."""
    try:
        key = score.analyze("key")
    except Exception:
        return 0
    target = 0 if key.mode == "major" else 9
    shift = (target - key.tonic.pitchClass) % 12
    return shift - 12 if shift > 6 else shift


def limit_polyphony(pitches: list[int]) -> tuple[int, ...]:
    """Keep the lowest note and the highest ones if a chord has too many notes."""
    if len(pitches) > MAX_POLYPHONY:
        pitches = [pitches[0]] + pitches[-(MAX_POLYPHONY - 1):]
    return tuple(pitches)


def extract_events(score, transpose: bool = True) -> list[tuple[tuple[int, ...], float]]:
    """Convert a music21 score into [(pitches, wait_until_next_event), ...]."""
    shift = key_shift(score) if transpose else 0
    onsets: dict[int, set[int]] = defaultdict(set)

    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            pitches = [element.pitch.midi]
        elif isinstance(element, chord.Chord):
            pitches = [p.midi for p in element.pitches]
        else:
            continue                          # drums and other unpitched sounds
        if element.tie is not None and element.tie.type in {"continue", "stop"}:
            continue                          # a tied note is a continuation, not a new note
        slot = int(round(float(element.offset) / GRID))
        onsets[slot].update(p + shift for p in pitches if 0 <= p + shift <= 127)

    slots = sorted(slot for slot in onsets if onsets[slot])
    events = []
    for here, following in zip(slots, slots[1:]):
        wait = min(MAX_DELTA, (following - here) * GRID)
        events.append((limit_polyphony(sorted(onsets[here])), wait))
    return events


def load_pieces(source: str, midi_dir: Path, max_files: int | None, transpose: bool):
    """Yield (name, events) for every usable piece."""
    midi_files = find_midi_files(midi_dir) if source in {"auto", "midi"} else []
    if source == "midi" and not midi_files:
        raise SystemExit(f"No .mid files found in {midi_dir}. Add some or use --source bach.")

    if midi_files:
        print(f"Reading {len(midi_files)} MIDI files from {midi_dir}")
        scores = ((path.name, path) for path in midi_files)
        parse = lambda path: converter.parse(str(path))             # noqa: E731
    else:
        print("No MIDI files in data/midi - using the Bach chorales that come with music21")
        chorale_files = list(corpus.chorales.Iterator(returnType="filename"))
        scores = ((fn, fn) for fn in chorale_files)
        parse = lambda fn: corpus.parse(fn)                         # noqa: E731

    used = 0
    for index, (name, item) in enumerate(scores, start=1):
        if max_files and used >= max_files:
            break
        try:
            events = extract_events(parse(item), transpose)
        except Exception as error:                                   # a broken file should not stop the run
            print(f"  skipped {name}: {error}")
            continue
        if len(events) > SEQ_LEN + 1:
            used += 1
            yield name, events
        if index % 25 == 0:
            print(f"  {index} files read ({used} usable)")


# ---------------------------------------------------------- building the data
def simplify(chord_notes, known):
    """Replace a rare chord by its top note, or drop it (None) if that is rare too."""
    if chord_notes in known:
        return chord_notes
    top = (chord_notes[-1],)
    return top if top in known else None


def build_vocab(pieces: list[list]) -> dict:
    counts = Counter(c for events in pieces for c, _ in events)
    common = {c for c, n in counts.items() if n >= MIN_CHORD_COUNT}
    chords = sorted(common)
    return {"chords": [list(c) for c in chords], "durations": DURATIONS,
            "seq_len": SEQ_LEN, "grid": GRID}


def duration_id(wait: float) -> int:
    return min(len(DURATIONS) - 1, max(0, int(round(wait / GRID)) - 1))


def encode_events(events, chord_to_id: dict) -> tuple[list[int], list[int]]:
    """Turn events into two lists of integer ids (chords, waits)."""
    known = set(chord_to_id)
    chord_ids, duration_ids = [], []
    for pitches, wait in events:
        pitches = simplify(tuple(pitches), known)
        if pitches is None:
            continue
        chord_ids.append(chord_to_id[pitches])
        duration_ids.append(duration_id(wait))
    return chord_ids, duration_ids


def make_windows(chord_ids, duration_ids, piece_index, rows):
    """Cut one piece into overlapping windows; the target is the input shifted by one."""
    for start in range(0, len(chord_ids) - SEQ_LEN, STRIDE):
        stop = start + SEQ_LEN
        rows["chord_in"].append(chord_ids[start:stop])
        rows["dur_in"].append(duration_ids[start:stop])
        rows["chord_out"].append(chord_ids[start + 1:stop + 1])
        rows["dur_out"].append(duration_ids[start + 1:stop + 1])
        rows["piece_id"].append(piece_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "midi", "bach"], default="auto")
    parser.add_argument("--midi-dir", type=Path, default=MIDI_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="stop after this many usable pieces")
    parser.add_argument("--no-transpose", action="store_true", help="keep the original keys")
    args = parser.parse_args()

    pieces = [events for _, events in load_pieces(args.source, args.midi_dir, args.max_files, not args.no_transpose)]
    if not pieces:
        raise SystemExit("No usable pieces were found.")

    vocab = build_vocab(pieces)
    chord_to_id = {tuple(c): i for i, c in enumerate(vocab["chords"])}

    rows: dict[str, list] = {key: [] for key in ("chord_in", "dur_in", "chord_out", "dur_out", "piece_id")}
    total_events = 0
    for piece_index, events in enumerate(pieces):
        chord_ids, duration_ids = encode_events(events, chord_to_id)
        total_events += len(chord_ids)
        make_windows(chord_ids, duration_ids, piece_index, rows)

    arrays = {key: np.array(value, dtype=np.int32) for key, value in rows.items()}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_DIR / "sequences.npz", **arrays)
    (PROCESSED_DIR / "vocab.json").write_text(json.dumps(vocab))

    print(f"\n{len(pieces)} pieces, {total_events:,} events")
    print(f"{len(vocab['chords'])} different chords, {len(DURATIONS)} different waits")
    print(f"{len(arrays['chord_in']):,} training sequences of {SEQ_LEN} events")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

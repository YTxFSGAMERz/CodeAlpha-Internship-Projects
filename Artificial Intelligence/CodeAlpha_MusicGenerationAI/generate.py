"""Step 3 - compose new music with the trained model.

    python generate.py                                 # one piece of 200 events
    python generate.py --length 400 --temperature 1.0  # longer and more adventurous
    python generate.py --count 3 --wav                 # three pieces, each also saved as audio
    python generate.py --seed-midi my_song.mid         # start from the opening of your own MIDI file

How it works: the model gets a short starting phrase (the "seed"), predicts the next
chord and wait as probabilities, one is drawn at random, appended to the phrase,
and the loop repeats. A low temperature picks safe, predictable notes; a high one
takes more risks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np

from config import MODEL_DIR, OUTPUT_DIR, SEQ_LEN
from music_utils import events_to_notes, save_midi, save_wav, synthesize


def sample_index(probabilities, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    """Draw an index from a probability list.

    temperature < 1 makes likely choices even more likely, > 1 flattens the odds.
    top_k keeps only the k most likely choices, which avoids very odd notes.
    """
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None))
    logits /= max(temperature, 1e-3)
    if 0 < top_k < len(logits):
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    weights = np.exp(logits - logits.max())
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def seed_from_midi(path: Path, chord_to_id: dict) -> tuple[list[int], list[int]]:
    from music21 import converter
    from preprocess import encode_events, extract_events

    chords, waits = encode_events(extract_events(converter.parse(str(path))), chord_to_id)
    if len(chords) < SEQ_LEN:
        raise SystemExit(f"{path.name} has only {len(chords)} usable events; at least {SEQ_LEN} are needed.")
    return chords[:SEQ_LEN], waits[:SEQ_LEN]


def generate_events(model, seed_chords, seed_waits, length, temperature, top_k, rng):
    """Continue the seed for `length` events. Returns lists of chord ids and wait ids."""
    chords, waits = list(seed_chords), list(seed_waits)
    for _ in range(length):
        window = {
            "chord_in": np.array([chords[-SEQ_LEN:]], dtype="int32"),
            "dur_in": np.array([waits[-SEQ_LEN:]], dtype="int32"),
        }
        prediction = model(window, training=False)
        next_chord = np.asarray(prediction["chord_out"])[0, -1]
        next_wait = np.asarray(prediction["dur_out"])[0, -1]
        chords.append(sample_index(next_chord, temperature, top_k, rng))
        waits.append(sample_index(next_wait, temperature, min(top_k, 5) if top_k else 0, rng))
    return chords[len(seed_chords):], waits[len(seed_waits):]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length", type=int, default=200, help="number of events to generate")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=12, help="only choose among the k likeliest chords (0 = all)")
    parser.add_argument("--bpm", type=float, default=100, help="playback speed")
    parser.add_argument("--count", type=int, default=1, help="how many pieces to create")
    parser.add_argument("--seed-midi", type=Path, help="start from the opening of this MIDI file")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for repeatable results")
    parser.add_argument("--output", type=Path, help="output .mid path (only with --count 1)")
    parser.add_argument("--wav", action="store_true", help="also save each piece as a WAV audio file")
    args = parser.parse_args()

    model_file = MODEL_DIR / "music_lstm.keras"
    if not model_file.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")
    model = keras.models.load_model(model_file)
    vocab = json.loads((MODEL_DIR / "vocab.json").read_text())
    chord_to_id = {tuple(c): i for i, c in enumerate(vocab["chords"])}
    rng = np.random.default_rng(args.seed)

    seeds = None if args.seed_midi else np.load(MODEL_DIR / "seeds.npz")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for number in range(1, args.count + 1):
        if args.seed_midi:
            seed_chords, seed_waits = seed_from_midi(args.seed_midi, chord_to_id)
        else:
            pick = int(rng.integers(len(seeds["chord_in"])))
            seed_chords, seed_waits = seeds["chord_in"][pick].tolist(), seeds["dur_in"][pick].tolist()

        chord_ids, wait_ids = generate_events(model, seed_chords, seed_waits, args.length,
                                              args.temperature, args.top_k, rng)
        events = [(tuple(vocab["chords"][c]), vocab["durations"][w]) for c, w in zip(chord_ids, wait_ids)]

        if args.output and args.count == 1:
            midi_path = args.output
        else:
            midi_path = OUTPUT_DIR / f"generated_{stamp}_{number}.mid"
        save_midi(events, midi_path, args.bpm)

        pitches = [p for chord_notes, _ in events for p in chord_notes]
        seconds = sum(wait for _, wait in events) * 60 / args.bpm
        print(f"[{number}/{args.count}] {midi_path}  ({len(events)} events, {seconds:.0f} s, "
              f"notes from {min(pitches)} to {max(pitches)})")

        if args.wav:
            wav_path = save_wav(synthesize(events_to_notes(events, args.bpm)), Path(midi_path).with_suffix(".wav"))
            print(f"        audio: {wav_path}")


if __name__ == "__main__":
    main()

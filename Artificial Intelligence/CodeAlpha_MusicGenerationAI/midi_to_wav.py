"""Turn any MIDI file into a WAV audio file (no extra software needed).

    python midi_to_wav.py outputs/sample_generated.mid
    python midi_to_wav.py song.mid song.wav
"""

import argparse
from pathlib import Path

from music21 import chord, converter, note

from music_utils import save_wav, synthesize


def midi_to_notes(path: Path) -> list[tuple[float, float, int]]:
    """Read a MIDI file as [(start_seconds, length_seconds, midi_pitch), ...]."""
    score = converter.parse(str(path))
    notes = []
    for item in score.flatten().secondsMap:
        element = item["element"]
        if isinstance(element, note.Note):
            pitches = [element.pitch.midi]
        elif isinstance(element, chord.Chord):
            pitches = [p.midi for p in element.pitches]
        else:
            continue
        for pitch in pitches:
            notes.append((item["offsetSeconds"], item["durationSeconds"], pitch))
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("midi", type=Path)
    parser.add_argument("wav", type=Path, nargs="?", help="output file (default: same name with .wav)")
    args = parser.parse_args()

    notes = midi_to_notes(args.midi)
    if not notes:
        raise SystemExit("That MIDI file has no pitched notes.")
    target = args.wav or args.midi.with_suffix(".wav")
    save_wav(synthesize(notes), target)
    print(f"Wrote {target} ({len(notes)} notes)")


if __name__ == "__main__":
    main()

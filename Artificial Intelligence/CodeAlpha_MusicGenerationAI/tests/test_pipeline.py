"""Tests for the music pipeline. Run with: pytest"""

import sys
import wave
from pathlib import Path

import numpy as np
from music21 import chord, converter, note, stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocess  # noqa: E402
from config import DURATIONS, MAX_DELTA, SEQ_LEN, STRIDE  # noqa: E402
from generate import sample_index  # noqa: E402
from music_utils import events_to_notes, save_midi, save_wav, synthesize  # noqa: E402


def melody(pitches, length=1.0):
    piece = stream.Stream()
    for i, name in enumerate(pitches):
        piece.insert(i * length, note.Note(name, quarterLength=length))
    return piece


# ------------------------------------------------------------ reading music
def test_notes_become_events_with_the_wait_until_the_next_note():
    events = preprocess.extract_events(melody(["C4", "D4", "E4", "F4"]), transpose=False)
    assert events == [((60,), 1.0), ((62,), 1.0), ((64,), 1.0)]      # the last note has no "next" and is dropped


def test_chords_and_gaps_are_kept():
    piece = stream.Stream()
    piece.insert(0, chord.Chord(["C4", "E4", "G4"], quarterLength=1))
    piece.insert(3, note.Note("D4", quarterLength=1))                # two beats of silence before it
    piece.insert(4, note.Note("E4", quarterLength=1))
    assert preprocess.extract_events(piece, transpose=False) == [((60, 64, 67), 3.0), ((62,), 1.0)]


def test_a_long_silence_is_capped_at_one_bar():
    piece = stream.Stream()
    piece.insert(0, note.Note("C4"))
    piece.insert(20, note.Note("D4"))
    piece.insert(21, note.Note("E4"))
    assert preprocess.extract_events(piece, transpose=False)[0] == ((60,), MAX_DELTA)


def test_pieces_are_moved_to_c_major():
    d_major = ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"] * 3
    events = preprocess.extract_events(melody(d_major), transpose=True)
    assert events[0][0] == (60,)                                      # D became C
    assert {p % 12 for pitches, _ in events for p in pitches} <= {0, 2, 4, 5, 7, 9, 11}


def test_too_many_notes_in_a_chord_are_reduced():
    assert preprocess.limit_polyphony([40, 50, 60, 70, 80, 90]) == (40, 70, 80, 90)
    assert preprocess.limit_polyphony([60, 64, 67]) == (60, 64, 67)


# ------------------------------------------------------------- making the data
def test_rare_chords_are_simplified_or_dropped():
    known = {(60, 64, 67), (72,)}
    assert preprocess.simplify((60, 64, 67), known) == (60, 64, 67)
    assert preprocess.simplify((60, 64, 72), known) == (72,)         # falls back to its top note
    assert preprocess.simplify((50, 53), known) is None


def test_events_are_encoded_as_ids():
    chord_to_id = {(60,): 0, (62,): 1}
    events = [((60,), 0.25), ((62,), 1.0), ((99,), 1.0), ((60,), 9.0)]
    chords, waits = preprocess.encode_events(events, chord_to_id)
    assert chords == [0, 1, 0]                                        # the unknown note 99 was dropped
    assert waits == [0, 3, len(DURATIONS) - 1]                        # 0.25 -> 0, 1.0 -> 3, too long -> last


def test_windows_are_shifted_by_one_event():
    rows = {key: [] for key in ("chord_in", "dur_in", "chord_out", "dur_out", "piece_id")}
    ids = list(range(SEQ_LEN + 8))
    preprocess.make_windows(ids, ids, 0, rows)
    assert len(rows["chord_in"]) == len(range(0, len(ids) - SEQ_LEN, STRIDE))
    first_in, first_out = rows["chord_in"][0], rows["chord_out"][0]
    assert len(first_in) == SEQ_LEN and first_out[:-1] == first_in[1:]


# ----------------------------------------------------------------- generating
def test_sampling_follows_the_probabilities():
    rng = np.random.default_rng(0)
    probabilities = [0.1, 0.9]
    draws = [sample_index(probabilities, 1.0, 0, rng) for _ in range(3000)]
    assert 0.85 < np.mean(draws) < 0.95
    assert all(sample_index([0.2, 0.5, 0.3], 1.0, 1, rng) == 1 for _ in range(50))     # top_k=1 -> always the best
    assert all(sample_index([0.2, 0.5, 0.3], 0.01, 0, rng) == 1 for _ in range(50))    # near-zero temperature too


# ------------------------------------------------------------------ MIDI, audio
def test_events_are_converted_to_timed_notes():
    notes = events_to_notes([((60,), 1.0), ((62, 64), 0.5)], bpm=120)
    assert notes == [(0.0, 0.5, 60), (0.5, 0.25, 62), (0.5, 0.25, 64)]


def test_midi_file_can_be_read_back(tmp_path):
    events = [((60,), 1.0), ((62, 65, 69), 0.5), ((64,), 2.0)]
    path = save_midi(events, tmp_path / "out.mid", bpm=90)
    elements = list(converter.parse(str(path)).flatten().notes)
    assert len(elements) == 3
    assert [p.midi for p in elements[1].pitches] == [62, 65, 69]


def test_audio_is_rendered_and_saved(tmp_path):
    audio = synthesize(events_to_notes([((60,), 1.0), ((67,), 1.0)], bpm=120))
    assert audio.dtype == np.float32
    assert len(audio) > 22050 * 1.0                                   # at least the two notes' length
    assert 0.5 < np.abs(audio).max() <= 1.0                           # audible, not clipped

    path = save_wav(audio, tmp_path / "out.wav")
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 22050, len(audio))


# ---------------------------------------------------------------- the network
def test_model_outputs_a_probability_for_every_chord_and_wait():
    from model import build_model

    model = build_model(n_chords=50, n_durations=16, units=16, chord_embed=8, dur_embed=4)
    window = {"chord_in": np.random.randint(0, 50, (3, SEQ_LEN)), "dur_in": np.random.randint(0, 16, (3, SEQ_LEN))}
    out = model(window, training=False)
    assert tuple(out["chord_out"].shape) == (3, SEQ_LEN, 50)
    assert tuple(out["dur_out"].shape) == (3, SEQ_LEN, 16)
    import keras
    chord_out_np = keras.ops.convert_to_numpy(out["chord_out"])
    assert np.allclose(chord_out_np.sum(axis=-1), 1.0, atol=1e-4)

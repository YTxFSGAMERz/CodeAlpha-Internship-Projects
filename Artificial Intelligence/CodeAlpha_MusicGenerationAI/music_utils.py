"""Helpers that turn generated events into a MIDI file and into audio.

An event is (pitches, wait): "play these MIDI note numbers now, then wait `wait`
quarter notes before the next event" - the same format preprocess.py produces.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream, tempo

SAMPLE_RATE = 22050
HARMONICS = [1.0, 0.5, 0.28, 0.14, 0.07]   # overtone strengths: a soft, piano-like tone


def events_to_notes(events, bpm: float) -> list[tuple[float, float, int]]:
    """[(pitches, wait)] -> [(start_seconds, length_seconds, midi_pitch)]."""
    seconds_per_beat = 60.0 / bpm
    notes, beat = [], 0.0
    for pitches, wait in events:
        for pitch in pitches:
            notes.append((beat * seconds_per_beat, wait * seconds_per_beat, int(pitch)))
        beat += wait
    return notes


def save_midi(events, path: str | Path, bpm: float = 100) -> Path:
    """Write the events to a MIDI file with a piano sound."""
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=bpm))
    beat = 0.0
    for pitches, wait in events:
        element = note.Note(int(pitches[0])) if len(pitches) == 1 else chord.Chord([int(p) for p in pitches])
        element.quarterLength = wait
        part.insert(beat, element)
        beat += wait

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.Score([part]).write("midi", fp=str(path))
    return path


def synthesize(notes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mix (start, length, pitch) notes into one mono waveform with simple additive synthesis."""
    if not notes:
        return np.zeros(1, dtype=np.float32)
    ring_out = 0.8                                     # seconds a note keeps fading after it ends
    total = max(start + length for start, length, _ in notes) + ring_out + 0.5
    audio = np.zeros(int(total * sample_rate) + 1)

    for start, length, pitch in notes:
        frequency = 440.0 * 2 ** ((pitch - 69) / 12)
        length = max(length, 0.12)
        t = np.arange(int((length + ring_out) * sample_rate)) / sample_rate

        tone = sum(
            strength * np.sin(2 * np.pi * frequency * k * t)
            for k, strength in enumerate(HARMONICS, start=1)
            if frequency * k < sample_rate / 2
        )
        decay = 1.4 + frequency / 600                  # higher notes fade faster
        envelope = np.minimum(t / 0.012, 1.0) * np.exp(-decay * t)
        envelope *= np.where(t > length, np.exp(-(t - length) * 9.0), 1.0)

        first = int(start * sample_rate)
        last = min(len(audio), first + len(t))
        audio[first:last] += (tone * envelope)[: last - first]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio *= 0.9 / peak
    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a mono 16-bit WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path

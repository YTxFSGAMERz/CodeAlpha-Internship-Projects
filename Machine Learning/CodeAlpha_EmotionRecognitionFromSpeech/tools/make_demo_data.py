"""
Creates a SYNTHETIC mini dataset that follows the RAVDESS file-naming convention, so you can
check that your installation and the whole pipeline work before downloading the real data.

The audio is NOT speech - it is generated tones whose pitch, movement, vibrato, loudness and
roughness differ per emotion - so the accuracy you get on it means nothing about real speech.
Use RAVDESS, TESS or EMO-DB for real results.

    python tools/make_demo_data.py
    python train.py --ravdess demo_data/RAVDESS --epochs 15
"""
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
ROOT = Path(__file__).resolve().parents[1] / "demo_data" / "RAVDESS"

# RAVDESS emotion code -> (name, base pitch Hz, pitch movement Hz, vibrato Hz, loudness, noise level)
RECIPES = {
    "01": ("neutral", 120, 0, 0, 0.30, 0.00),
    "02": ("calm", 100, 10, 2, 0.20, 0.00),
    "03": ("happy", 220, 60, 5, 0.50, 0.01),
    "04": ("sad", 90, -25, 3, 0.15, 0.00),
    "05": ("angry", 180, 15, 12, 0.85, 0.05),
    "06": ("fear", 260, 30, 9, 0.40, 0.02),
    "07": ("disgust", 110, -10, 1, 0.35, 0.04),
    "08": ("surprise", 300, 120, 4, 0.55, 0.01),
}


def synth(rng, f0, move, vibrato, loud, noise, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    pitch = f0 * rng.uniform(0.85, 1.15) + move * t / seconds
    if vibrato:
        pitch = pitch + 8 * np.sin(2 * np.pi * vibrato * t)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    wave = sum(np.sin(h * phase) / h for h in range(1, 6))        # a few harmonics, like a voice
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.15)   # fade in and out
    wave = loud * envelope * wave / np.max(np.abs(wave))
    wave = wave + noise * rng.standard_normal(len(t))
    return np.concatenate([np.zeros(int(0.3 * SR)), wave, np.zeros(int(0.2 * SR))]).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    count = 0
    for actor in range(1, 9):
        folder = ROOT / f"Actor_{actor:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for code, (_, f0, move, vib, loud, noise) in RECIPES.items():
            for statement in (1, 2):
                for repetition in (1, 2):
                    name = f"03-01-{code}-01-{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                    sf.write(folder / name, synth(rng, f0, move, vib, loud, noise), SR, subtype="PCM_16")
                    count += 1
    print(f"Created {count} synthetic clips in {ROOT}")


if __name__ == "__main__":
    main()

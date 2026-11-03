"""The neural network: an LSTM that reads a window of events and predicts the next one.

Each event has two parts, so the network has two inputs and two outputs:

    chord id  ─► Embedding ─┐
                            ├─► LSTM ─► LSTM ─┬─► Dense softmax ─► next chord
    wait id   ─► Embedding ─┘                 └─► Dense softmax ─► next wait

At every position of the window it predicts the event that follows, which gives
the network SEQ_LEN learning signals per window instead of one.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers


def build_model(n_chords: int, n_durations: int, units: int = 192, chord_embed: int = 64,
                dur_embed: int = 8, dropout: float = 0.3, learning_rate: float = 2e-3) -> keras.Model:
    chord_in = keras.Input(shape=(None,), dtype="int32", name="chord_in")
    dur_in = keras.Input(shape=(None,), dtype="int32", name="dur_in")

    x = layers.Concatenate()([
        layers.Embedding(n_chords, chord_embed)(chord_in),
        layers.Embedding(n_durations, dur_embed)(dur_in),
    ])
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)

    outputs = {
        "chord_out": layers.Dense(n_chords, activation="softmax", name="chord_out")(x),
        "dur_out": layers.Dense(n_durations, activation="softmax", name="dur_out")(x),
    }
    model = keras.Model(inputs={"chord_in": chord_in, "dur_in": dur_in}, outputs=outputs, name="music_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={"chord_out": "sparse_categorical_crossentropy", "dur_out": "sparse_categorical_crossentropy"},
        loss_weights={"chord_out": 1.0, "dur_out": 0.5},
        metrics={"chord_out": ["accuracy"], "dur_out": ["accuracy"]},
    )
    return model

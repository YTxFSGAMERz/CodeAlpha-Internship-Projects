"""Three neural networks for speech emotion recognition - pick one with  train.py --model ..."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, models


def _conv_block(x, filters):
    x = layers.Conv1D(filters, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.Dropout(0.3)(x)


def build_model(kind: str, input_shape, n_classes: int) -> models.Model:
    """
    input_shape = (frames, features) = (130, 120)

    cnn       3 x Conv1D blocks, then average + max pooling over time
    lstm      2 stacked bidirectional LSTMs read the MFCC sequence directly
    cnn_lstm  Conv1D blocks find local patterns, a bidirectional LSTM models how they evolve
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    if kind in ("cnn", "cnn_lstm"):
        for filters in (64, 128, 128):
            x = _conv_block(x, filters)
    if kind == "cnn":
        x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    elif kind == "cnn_lstm":
        x = layers.Bidirectional(layers.LSTM(128))(x)
    elif kind == "lstm":
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
    else:
        raise ValueError(f"Unknown model '{kind}' - choose cnn, lstm or cnn_lstm")
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=f"emotion_{kind}")

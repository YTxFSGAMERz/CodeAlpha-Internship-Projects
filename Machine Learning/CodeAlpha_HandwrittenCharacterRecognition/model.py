"""The CNN used for MNIST / EMNIST character recognition."""
from __future__ import annotations

from tensorflow.keras import layers, models


def conv_block(x, filters):
    """Conv -> BatchNorm -> ReLU. BatchNorm makes training faster and more stable."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def build_cnn(num_classes: int, input_shape=(28, 28, 1), augment: bool = True) -> models.Model:
    """
    Input is the RAW pixel image (0-255); rescaling happens inside the model.

        [random rotate / shift / zoom]   (training only - a cheap way to get more data)
        2 x Conv32 -> MaxPool -> Dropout
        2 x Conv64 -> MaxPool -> Dropout
        Flatten -> Dense256 -> Dropout -> softmax
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    if augment:  # these layers are automatically switched off at prediction time
        x = layers.RandomRotation(0.04, fill_mode="constant")(x)                     # about +-14 degrees
        x = layers.RandomTranslation(0.08, 0.08, fill_mode="constant")(x)            # +-8 % of the image
        x = layers.RandomZoom(0.1, fill_mode="constant")(x)

    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="char_cnn")

import os

# Default to torch backend for Keras 3 when TensorFlow is not present
os.environ.setdefault("KERAS_BACKEND", "torch")

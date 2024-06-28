"""
Keras Code for Chinook Depth Model
"""

# pylint: disable=invalid-name, import-error, no-name-in-module

import os

from functools import partial
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate


# pylint: disable=redefined-builtin
def split_data(N, features, data):
    """
    Inputs:
    - N: int, number of choices
    - features: list of strings, names of features
    - data: dict, data loaded from tfrecord file

    Splits the data into inputs and labels for the model
    """
    inputs = {}
    for i in range(N):
        input = tf.stack(
            [tf.cast(data[f"{feature}_{i}"], tf.float32) for feature in features]
        )
        inputs[f"input_{i}"] = input
    label = to_categorical(data["_selected"], num_classes=N)
    return inputs, label


def load_data(data_dir, N, features, batch_size, shuffle_buffer_size):
    """
    Inputs:
    - data_dir: str, path to directory containing tfrecord files
    - N: int, number of choices
    - features: list of strings, names of features
    - batch_size: int, batch size
    - shuffle_buffer_size: int, size of buffer for shuffling data

    Returns a tf.data.Dataset object containing the data
    """
    feature_description = {
        "_selected": tf.io.FixedLenFeature([], tf.int64),
    }
    for i in range(N):
        for feature in features:
            feature_description[f"{feature}_{i}"] = tf.io.FixedLenFeature(
                [], tf.float32
            )

    def _parse_function(proto):
        return tf.io.parse_single_example(proto, feature_description)

    tfrecord_files = []
    for path in os.listdir(data_dir):
        if path.endswith(".tfrecord"):
            tfrecord_files.append(os.path.join(data_dir, path))

    data = tf.data.TFRecordDataset(tfrecord_files)
    data = data.map(_parse_function)
    data = data.map(partial(split_data, N, features))
    data = data.shuffle(buffer_size=shuffle_buffer_size)
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


def build_model(N, features, layers, final_activation="linear"):
    """
    Inputs:
    - N: int, number of choices
    - features: list of strings, names of features
    - layers: list of keras layers
    - final_activation: str, activation function for final layer

    Returns a keras model
    """
    layers.append(Dense(1, activation=final_activation))
    inputs = [Input(shape=(len(features),), name=f"input_{i}") for i in range(N)]
    outcomes = []
    for input in inputs:
        last_layer = input
        for layer in layers:
            last_layer = layer(last_layer)
        outcomes.append(last_layer)

    outcomes = concatenate(outcomes)

    output_layer = Dense(N, activation="softmax")
    output = output_layer(outcomes)
    output_layer.set_weights([np.eye(N), np.zeros(N)])
    output_layer.trainable = False

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model, layers

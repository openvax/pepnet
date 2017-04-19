# Copyright (c) 2017. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.layers import (
    Input,
    Embedding,
    Dropout,
    SpatialDropout1D,
    Dense,
    Activation,
    BatchNormalization,
    Concatenate,
    Multiply,
    Add,
    Flatten,
    Lambda,
    LSTM,
    GRU,
    Bidirectional
)
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import (
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D
)
import keras.backend as K
import keras.initializers

def make_onehot_sequence_input(name, length, n_symbols):
    return Input(
            shape=(length, n_symbols),
            name=name,
            dtype="float32")

def make_index_sequence_input(name, length):
    return Input(
        shape=(length,),
        dtype="int32",
        name=name)

def make_sequence_input(name, length, n_symbols, encoding):
    assert encoding in {"onehot", "index"}
    if encoding == "index":
        return make_index_sequence_input(name, length)
    else:
        return make_onehot_sequence_input(name, length, n_symbols)

def make_numeric_input(name, dim, dtype):
    return Input(name=name, shape=(dim,), dtype=dtype)


def merge(values, merge_mode):
    assert merge_mode in {"concat", "add", "multiply"}
    if len(values) == 1:
        return values[0]
    elif merge_mode == "concat":
        return Concatenate()(values)
    elif merge_mode == "add":
        return Add()(values)
    elif merge_mode == "multiply":
        return Multiply()(values)

def flatten(value):
    return Flatten()(value)

def regularize(value, batch_normalization=False, dropout=0.0):
    if batch_normalization:
        value = BatchNormalization()(value)
    if dropout:
        value = Dropout(dropout)(value)
    return value

def embedding(
        value, n_symbols, output_dim, dropout=0, initial_weights=None):
    if initial_weights:
        n_rows, n_cols = initial_weights.shape
        if n_rows != n_symbols or n_cols != output_dim:
            raise ValueError(
                "Wrong shape for embedding: expected (%d, %d) but got "
                "(%d, %d)" % (
                    n_symbols, output_dim,
                    n_rows, n_cols))
        embedding_layer = Embedding(
            input_dim=n_symbols,
            output_dim=output_dim,
            mask_zero=False,
            weights=[initial_weights])
    else:
        embedding_layer = Embedding(
            input_dim=n_symbols,
            output_dim=output_dim,
            mask_zero=False)

    value = embedding_layer(value)

    if dropout:
        value = SpatialDropout1D(dropout)(value)

    return value

def conv(value, filter_size, output_dim, dropout=0.0, padding="valid"):
    """
    Perform a single scale of convolution and optionally add spatial dropout.
    """
    conv_layer = Conv1D(
            filters=output_dim,
            kernel_size=filter_size,
            padding=padding)
    convolved = conv_layer(value)
    if dropout:
        # random drop some of the convolutional filters
        convolved = SpatialDropout1D(dropout)(convolved)
    return convolved

def aligned_convolutions(value, filter_sizes, output_dim, dropout=0.0):
    """
    Perform convolutions at multiple scales and concatenate their outputs.

    Parameters
    ----------
    filter_sizes : int or list of int
        Widths of convolutional filters

    output_dim : int
        Number of filters per width

    dropout : float
        Dropout after convolutional
    """
    if isinstance(filter_sizes, int):
        filter_sizes = [filter_sizes]

    convolved_list = []
    for size in filter_sizes:
        convolved_list.append(conv(
            value=value,
            filter_size=size,
            output_dim=output_dim,
            dropout=dropout,
            padding="same"))
    convolved = merge(convolved_list, "concat")
    return convolved

def local_max_pooling(value, size=3, stride=2):
    return MaxPooling1D(pool_size=size, strides=stride)(value)

def global_max_pooling(value, batch_normalization=False, dropout=0):
    return regularize(
        value=GlobalMaxPooling1D()(value),
        batch_normalization=batch_normalization,
        dropout=dropout)

def global_mean_pooling(value, batch_normalization=False, dropout=0):
    return regularize(
        value=GlobalAveragePooling1D()(value),
        batch_normalization=batch_normalization,
        dropout=dropout)

def global_max_and_mean_pooling(
        value, batch_normalization=False, dropout=0):
    return merge([
        global_max_pooling(
            value=value,
            batch_normalization=batch_normalization,
            dropout=dropout),
        global_mean_pooling(
            value=value,
            batch_normalization=batch_normalization,
            dropout=dropout)], "concat")

def dense(value, dim, activation, init="glorot_uniform", name=None):
    if name:
        # hidden layer fully connected layer
        value = Dense(
            units=dim, kernel_initializer=init, name="%s_dense" % name)(value)
        value = Activation(activation, name=name)(value)
    else:
        value = Dense(units=dim, kernel_initializer=init)(value)
        value = Activation(activation)(value)
    return value

def dense_layers(
        value,
        layer_sizes,
        activation="relu",
        init="glorot_uniform",
        batch_normalization=False,
        dropout=0.0):
    for i, dim in enumerate(layer_sizes):
        value = regularize(
            value=dense(value, dim=dim, init=init, activation=activation),
            batch_normalization=batch_normalization,
            dropout=dropout)
    return value


def highway_layer(value, activation="tanh", gate_bias=-3):
    dims = K.int_shape(value)
    if len(dims) != 2:
        raise ValueError(
            "Expected 2d value (batch_size, dims) but got shape %s" % (
                dims,))
    dim = dims[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
    gate = Activation("sigmoid")(gate)
    negated_gate = Lambda(
        lambda x: 1.0 - x,
        output_shape=(dim,))(gate)
    transformed = Dense(units=dim)(value)
    transformed = Activation(activation)(value)
    transformed_gated = Multiply()([gate, transformed])
    pass_through_gated = Multiply()([negated_gate, value])
    value = Add()([transformed_gated, pass_through_gated])
    return value

def highway_layers(
        value,
        n_layers,
        activation="tanh",
        batch_normalization=False,
        dropout=0):
    """
    Construct "highway" layers which default to the identity function
    but can learn to transform their input. The batch normalization
    and dropout parameters only affect the output of the last layer.
    """
    for i in range(n_layers):
        value = highway_layer(value, activation=activation, dropout=dropout)
    return regularize(
        value=value,
        batch_normalization=batch_normalization,
        dropout=dropout)

def recurrent_layers(
        value,
        layer_sizes,
        bidirectional=True,
        dropout=0.0,
        rnn_type="lstm"):
    """
    Make one or more RNN layers
    """
    if rnn_type == "lstm":
        rnn_class = LSTM
    elif rnn_type == "gru":
        rnn_class = GRU
    else:
        raise ValueError("Unknown RNN type: %s" % (rnn_type,))

    for i, layer_size in enumerate(layer_sizes):
        last_layer = (i == len(layer_sizes) - 1)
        rnn_layer = rnn_class(
            layer_size,
            return_sequences=not last_layer,
            dropout=dropout)
        if bidirectional:
            rnn_layer = Bidirectional(rnn_layer, merge_mode="concat")
        value = rnn_layer(value)
    return value

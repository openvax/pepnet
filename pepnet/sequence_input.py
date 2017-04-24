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

from serializable import Serializable

from .helpers import (
    aligned_convolutions,
    embedding,
    make_sequence_input,
    local_max_pooling,
    global_max_and_mean_pooling,
    flatten,
    recurrent_layers,
    highway_layers,
    dense_layers)
from .encoder import Encoder

class SequenceInput(Serializable):
    def __init__(
            self,
            length,
            name=None,
            variable_length=False,
            n_symbols=None,
            # embedding of symbol indices into vectors
            encoding="index",
            embedding_dim=32,
            embedding_dropout=0,
            embedding_mask_zero=True,
            # convolutional layers
            conv_filter_sizes=[],
            n_conv_layers=1,
            conv_output_dim=16,
            conv_dropout=0,
            conv_activation="linear",
            conv_weight_source=None,
            pool_size=3,
            pool_stride=2,
            # RNN
            rnn_layer_sizes=[],
            rnn_type="lstm",
            rnn_bidirectional=True,
            # global pooling of conv or RNN outputs
            global_pooling=False,
            global_pooling_batch_normalization=False,
            global_pooling_dropout=0,
            # transform global pooling representation with
            # dense layers
            dense_layer_sizes=[],
            dense_activation="relu",
            dense_dropout=0.25,
            dense_batch_normalization=False,
            # highway layers
            n_highway_layers=0,
            highway_activation="tanh",
            highway_batch_normalization=False,
            highway_dropout=0,):
        """
        Parameters
        ----------
        length : int
            Maximum length of sequence

        name : str
            Name of input sequence

        variable_length : bool
            Do we expect padding '-' characters in the input?

        n_symbols : int
            Number of distinct symbols in sequences, default expects
            20 amino acids + 1 character for padding ('-')

        encoding : {"index", "onehot"}
            How are symbols represented: via integer indices or boolean
            vectors?

        embedding_dim : int
            How many dimensions in the symbol embedding
            (only used for index encoding)

        embedding_dropout : float
            What fraction of symbol representations are randomly set to 0
            during training?

        embedding_mask_zero : bool
            Mask zero values in the input sequences

        conv_filter_sizes : list of int
            Width of convolutional filters to apply to input vectors

        n_conv_layers : int
            Number of convolutional layers (with interleaving max pooling)
            to create

        conv_output_dim : int
            Number of filters per size of convolution

        conv_dropout : float
            Fraction of convolutional activations to randomly set to 0 during
            training

        conv_weight_source : tensor, optional
            Determine weights of the convolution as a function of this
            tensor (rather than learning parameters separately)

        pool_size : int
            If using more than one convolutional layer, how many timesteps
            from a previous layer get combined via maxpooling before the next
            layer.

        pool_stride : int
            If using more than one convolutional layer, stride of the max
            pooling.

        rnn_layer_sizes : list of int
            Dimensionality of each RNN layer, defaults no RNN layers

        rnn_type : {"lstm", "gru"}
            Whether to use GRU or LSTM models for each RNN layer

        rnn_bidirectional : bool
            Use bidirectional RNNs

        global_pooling : bool
            Pool (mean & max) activations across sequence length

        global_pooling_batch_normalization : bool
            Apply BatchNormalization after global pooling

        global_pooling_dropout : float
            Fraction of entries to randomly drop during training after
            global pooling

        dense_layer_sizes : list of int
            Dimensionality of dense transformations after convolutional
            and recurrent layers

        dense_activation: str
            Activation function to use after each dense layer

        dense_dropout : float
            Fraction of dense output values to drop during training

        dense_batch_normalization : bool
            Apply batch normalization between hidden layers

        n_highway_layers : int
            Number of highway layers to use after dense layers

        highway_batch_normalization : bool
            Apply BatchNormalization after final highway layer

        highway_dropout : float
            Apply dropout after final highway layer

        highway_activation : str
            Activation function of each layer in the highway network
        """
        self.name = name
        self.length = length

        if encoding not in {"index", "onehot"}:
            raise ValueError("Invalid encoding: %s" % encoding)
        self.encoding = encoding
        self.variable_length = variable_length

        if not n_symbols:
            if variable_length:
                n_symbols = 21
            else:
                n_symbols = 20

        self.n_symbols = n_symbols
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.embedding_mask_zero = embedding_mask_zero

        if isinstance(conv_filter_sizes, int):
            conv_filter_sizes = [conv_filter_sizes]

        self.conv_filter_sizes = conv_filter_sizes
        self.conv_dropout = conv_dropout
        self.conv_output_dim = conv_output_dim
        self.conv_activation = conv_activation
        self.n_conv_layers = n_conv_layers
        self.conv_weight_source = conv_weight_source
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        if isinstance(rnn_layer_sizes, int):
            rnn_layer_sizes = [rnn_layer_sizes]
        self.rnn_layer_sizes = rnn_layer_sizes
        self.rnn_type = rnn_type
        self.rnn_bidirectional = rnn_bidirectional

        self.global_pooling = global_pooling
        self.global_pooling_batch_normalization = global_pooling_batch_normalization
        self.global_pooling_dropout = global_pooling_dropout

        # Dense layers after all temporal processing
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.dense_batch_normalization = dense_batch_normalization

        # Highway network after dense layers
        self.n_highway_layers = n_highway_layers
        self.highway_batch_normalization = highway_batch_normalization
        self.highway_dropout = highway_dropout
        self.highway_activation = highway_activation

    def build(self):
        input_object = make_sequence_input(
            encoding=self.encoding,
            name=self.name,
            length=self.length,
            n_symbols=self.n_symbols)

        if self.encoding == "index":
            assert self.embedding_dim > 0, \
                "Invalid embedding dim: %d" % self.embedding_dim
            value = embedding(
                input_object,
                n_symbols=self.n_symbols,
                output_dim=self.embedding_dim,
                mask_zero=self.embedding_mask_zero,
                dropout=self.embedding_dropout)
        else:
            value = input_object

        if self.conv_filter_sizes:
            conv_weight_source = self.conv_weight_source
            if conv_weight_source is not None and isinstance(
                    conv_weight_source, SequenceInput):
                conv_weight_source = conv_weight_source.build()[1]
            for i in range(self.n_conv_layers):
                value = aligned_convolutions(
                    value,
                    filter_sizes=self.conv_filter_sizes,
                    output_dim=self.conv_output_dim,
                    dropout=self.conv_dropout,
                    activation=self.conv_activation,
                    weight_source=conv_weight_source)
                # add max pooling for all layers before the last
                if i + 1 < self.n_conv_layers:
                    value = local_max_pooling(
                        value,
                        size=self.pool_size,
                        stride=self.pool_stride)

        if len(self.rnn_layer_sizes) > 0:
            value = recurrent_layers(
                value=value,
                layer_sizes=self.rnn_layer_sizes,
                rnn_type=self.rnn_type,
                bidirectional=self.rnn_bidirectional)

        if self.global_pooling:
            value = global_max_and_mean_pooling(
                value,
                batch_normalization=self.global_pooling_batch_normalization,
                dropout=self.global_pooling_dropout)

        if value.ndim > 2:
            value = flatten(value)

        value = dense_layers(
            value,
            layer_sizes=self.dense_layer_sizes,
            activation=self.dense_activation,
            dropout=self.dense_dropout,
            batch_normalization=self.dense_batch_normalization)

        if self.n_highway_layers:
            value = highway_layers(
                value,
                activation=self.highway_activation,
                n_layers=self.n_highway_layers,
                dropout=self.highway_dropout,
                batch_normalization=self.highway_batch_normalization)
        return input_object, value

    def encode(self, peptides):
        encoder = Encoder(variable_length_sequences=self.variable_length)
        if self.encoding == "index":
            return encoder.encode_index_array(
                peptides, max_peptide_length=self.length)
        else:
            return encoder.encode_onehot(
                peptides, max_peptide_length=self.length)

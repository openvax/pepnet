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

from .helpers import (
    aligned_convolutions,
    embedding,
    make_sequence_input,
    local_max_pooling,
    global_max_and_mean_pooling,
    flatten,
    recurrent_layers)
from .encoder import Encoder

class SequenceInput(object):
    def __init__(
            self,
            length,
            name=None,
            variable_length=False,
            n_symbols=None,
            encoding="index",
            embedding_dim=32,
            embedding_dropout=0,
            conv_filter_sizes=[],
            n_conv_layers=1,
            conv_output_dim=16,
            conv_dropout=0.1,
            pool_size=3,
            pool_stride=2,
            rnn_layer_sizes=[],
            rnn_type="lstm",
            rnn_bidirectional=True,
            rnn_dropout=0.0,
            global_pooling=False):
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

        rnn_dropout : float
            Recurrent dropout used in RNN layers

        global_pooling : bool
            Pool (mean & max) activations across sequence length
        """
        self.name = name
        self.length = length
        self.encoding = encoding
        self.variable_length = variable_length

        if not n_symbols:
            if variable_length:
                n_symbols = 21
            else:
                n_symbols = 20

        self.n_symbols = n_symbols
        self.embedding_dim = embedding_dim

        if isinstance(conv_filter_sizes, int):
            conv_filter_sizes = [conv_filter_sizes]

        self.conv_filter_sizes = conv_filter_sizes
        self.conv_dropout = conv_dropout
        self.conv_output_dim = conv_output_dim
        self.n_conv_layers = n_conv_layers
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        if isinstance(rnn_layer_sizes, int):
            rnn_layer_sizes = [rnn_layer_sizes]
        self.rnn_layer_sizes = rnn_layer_sizes
        self.rnn_type = rnn_type
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_dropout = rnn_dropout

        self.global_pooling = global_pooling

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
                output_dim=self.embedding_dim)
        else:
            value = input_object

        if self.conv_filter_sizes:
            for i in range(self.n_conv_layers):
                value = aligned_convolutions(
                    value,
                    filter_sizes=self.conv_filter_sizes,
                    output_dim=self.conv_output_dim,
                    dropout=self.conv_dropout)
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
                recurrent_dropout=self.rnn_dropout,
                bidirectional=self.rnn_bidirectional)

        if self.global_pooling:
            value = global_max_and_mean_pooling(value)

        if value.ndim > 2:
            value = flatten(value)

        return input_object, value

    def encode(self, peptides):
        encoder = Encoder(variable_length_sequences=self.variable_length)
        if self.encoding == "index":
            return encoder.encode_index_array(
                peptides, max_peptide_length=self.length)
        else:
            return encoder.encode_onehot(
                peptides, max_peptide_length=self.length)
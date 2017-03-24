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
    Dense, Embedding, Input, Flatten, Dropout, SpatialDropout1D)
from keras.layers.merge import concatenate
from keras.models import Model

from .keras_layers.masked_global_average_pooling import MaskedGlobalAveragePooling1D
from .keras_layers.masked_global_max_pooling import MaskedGlobalMaxPooling1D
from .keras_layers.drop_mask import DropMask

def make_variable_length_model_with_fixed_length_context(
        n_upstream=6,
        n_downstream=4,
        max_peptide_length=25,
        n_symbols=21,
        embedding_output_dim=32,
        embedding_dropout=0,
        dropout_before_hidden=0,
        dropout_after_hidden=0.25,
        n_hidden=20,
        hidden_activation="relu",
        optimizer="rmsprop",
        loss="mse",
        n_output=1,
        output_activation="sigmoid",
        compile_kwargs={}):
    """
    Make a model with three inputs:
        (1) upstream
            Fixed length amino acid sequence before the peptide
        (2) downstream
            Fixed length amino acid sequence after the peptide
        (3) peptide
            Variable number of amino acids

        Upstream --------------\
                                \
        Peptide  ---{Max/Avg}------> Hidden -> Output
                                /
        Downstream ------------/


    Parameters
    ----------
    n_upstream : int
        Number of amino acids in the upstream context

    n_downstream : int
        Number of amino acids in the downstream context

    max_peptide_length : int
        Longest peptide this model can consider

    n_symbols : int
        Number of characters we might encounter as part of a peptide or
        context sequence, can include '-' for padding of peptides

    embedding_output_dim : int
        Dimensionality of the amino acid embedding

    embedding_dropout : float
        Fraction of symbols to drop randomly during training

    dropout : float
        Fraction of activations to drop randomly during training

    n_hidden : int
        Dimensionality of hidden layer after merging of upstream/downstream/peptide

    hidden_activation : str
        Activation function for the hidden layer

    optimizer : str
        Optimization algorithm used to fit weights

    loss : str
        Target loss function

    n_output : int
        Number of outputs

    output_activation : str
        Activation function for outputs

    compile_kwargs : dict
        Extra keyword arguments to pass to model.compile

    """
    peptide_embedding = Embedding(
        input_dim=n_symbols + 1,
        output_dim=embedding_output_dim,
        mask_zero=True,
        name="peptide-embedding")
    # upstream
    upstream = Input(shape=(n_upstream,), dtype="int32", name="upstream")
    upstream_embedded = peptide_embedding(upstream)
    if embedding_dropout > 0:
        upstream_embedded = SpatialDropout1D(embedding_dropout)(upstream_embedded)

    # downstream
    downstream = Input(shape=(n_downstream,), dtype="int32", name="downstream")
    downstream_embedded = peptide_embedding(downstream)
    if embedding_dropout > 0:
        downstream_embedded = SpatialDropout1D(embedding_dropout)(downstream_embedded)

    # peptide
    peptide = Input(shape=(max_peptide_length,), dtype="int32", name="peptide")
    peptide_embedded = peptide_embedding(peptide)
    if embedding_dropout > 0:
        peptide_embedded = SpatialDropout1D(embedding_dropout)(peptide_embedded)

    peptide_max = MaskedGlobalMaxPooling1D()(peptide_embedded)
    peptide_avg = MaskedGlobalAveragePooling1D()(peptide_embedded)

    input_concat = concatenate([
        Flatten()(DropMask()(upstream_embedded)),
        Flatten()(DropMask()(downstream_embedded)),
        peptide_avg,
        peptide_max,
    ])
    if dropout_before_hidden:
        input_concat = Dropout(dropout_before_hidden)(input_concat)
    hidden = Dense(units=n_hidden, activation=hidden_activation, name="hidden")(input_concat)
    if dropout_after_hidden:
        hidden = Dropout(dropout_after_hidden)(hidden)
    output = Dense(units=n_output, activation=output_activation, name="output")(hidden)
    model = Model(
        inputs=[upstream, downstream, peptide],
        outputs=output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        **compile_kwargs)
    return model

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

from __future__ import (
    print_function,
    division,
    absolute_import,
)

import numpy as np
from six import integer_types
from six.moves import range

from .numeric import Numeric
from .nn_helpers import make_index_sequence_input, dense_layers, embedding, flatten

class DiscreteInput(Numeric):
    """
    Discrete inputs such as choosing an integer or from a predefined
    set of strings.
    """

    @classmethod
    def _check_choices(cls, choices):
        if isinstance(choices, integer_types):
            choices = list(range(choices))
        if isinstance(choices, set):
            choices = sorted(choices)
        if isinstance(choices, tuple):
            choices = list(choices)
        if isinstance(choices, np.ndarray):
            if len(choices.shape) != 1:
                raise ValueError("Expected sequence of choices, got array with shape %s" % (
                    choices.shape,))
        if not isinstance(choices, list):
            raise ValueError("Invalid type '%s' for choices" % (type(choices),))
        if len(choices) == 0:
            raise ValueError("Expected at least one element in choices")
        return choices

    @classmethod
    def _check_representation(cls, representation):
        original = representation
        normalized = representation.strip().lower()
        if normalized not in {"embedding", "onehot"}:
            raise ValueError(
                ("Invalid value '%s' for representation, "
                    "must be 'embedding' or 'onehot'") % original)
        return normalized

    @classmethod
    def _check_choice_type(cls, choices):
        expected_t = type(choices[0])
        for elt in choices:
            t = type(elt)
            if t is not expected_t:
                raise TypeError(
                    "Choices cannot contain elements of both types %s and %s"  % (
                        expected_t, t))
        return expected_t

    def __init__(
            self,
            choices,
            embedding_dim=32,
            name=None,
            dense_layer_sizes=[],
            dense_activation="relu",
            dense_dropout=0,
            dense_batch_normalization=False,
            dense_time_distributed=False,
            transform=None):
        """
        Parameters
        ----------
        choices : int, list of int, or list of str
            If an integer, then indicates that choices numbers 0,...,choices-1,
            otherwise expected list of all options.

        embedding_dim : int
            Number of dimensions to use for learned embedding representation
            of choices.

        name : str
            Name of input sequence

        representation : str
            - "embedding" means use an embedding layer to represent each choice
            as a learned vector

            - "onehot" means represent each choice by setting a corresponding
            position in a vector to 1, leaving all other entries as 0

        dense_layer_sizes : list of int
            Size of each dense layer after the input

        dense_activation : str
            Activation functin for dense layers after input

        dense_dropout : float
            Fraction of values to randomly set to 0 during training

        dense_batch_normalization : bool
            Use Batch Normalization after hidden layers

        dense_time_distributed : bool
            Apply dense layer to each input dimension separately

        transform : fn, optional
            Function to transform elements of numeric input/output
        """
        self.choices = self._check_choices(choices)
        self.choice_type = self._check_choice_type(self.choices)
        self.order_dict = {elt: i for (i, elt) in enumerate(self.choices)}
        self.n_choices = len(self.choices)
        self.embedding_dim = embedding_dim
        Numeric.__init__(
            self,
            name=name,
            dim=1,
            dense_layer_sizes=dense_layer_sizes,
            dense_activation=dense_activation,
            dense_dropout=dense_dropout,
            dense_batch_normalization=dense_batch_normalization,
            dense_time_distributed=dense_time_distributed,
            transform=transform)

    def encode(self, x):
        x = Numeric.encode(self, x)
        unique_elements = set(x)
        unsupported_elements = unique_elements.difference(self.choices)
        if len(unsupported_elements) > 0:
            raise ValueError("Input contained unsupported choices: %s" % (
                unsupported_elements,))
        n = len(x)
        index_array = np.zeros((n, 1), dtype="int32")
        d = self.order_dict
        for i, xi in enumerate(x):
            index_array[i, 0] = d[xi]
        return index_array

    def build(self):
        input_object = make_index_sequence_input(name=self.name, length=1)
        embedded = embedding(
            value=input_object,
            n_symbols=self.n_choices,
            output_dim=self.embedding_dim,
            dropout=0,
            initial_weights=None,
            mask_zero=False)
        hidden = dense_layers(
            value=embedded,
            layer_sizes=self.dense_layer_sizes,
            activation=self.dense_activation,
            dropout=self.dense_dropout,
            batch_normalization=self.dense_batch_normalization,
            time_distributed=self.dense_time_distributed)
        hidden = flatten(hidden)
        return input_object, hidden


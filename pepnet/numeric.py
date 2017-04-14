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

class Numeric(Serializable):
    """
    Base class for numeric input and output
    """
    def __init__(
            self,
            dim,
            name=None,
            dtype="float32",
            dense_layer_sizes=[],
            dense_activation="relu",
            dense_dropout=0,
            dense_batch_normalization=False,
            transform=None):
        """
        Parameters
        ----------
        dim : int
            Number of input dimensions

        name : str
            Name of input sequence

        dtype : str
            Most common option is "float32" but might also be "int32"

        dense_layer_sizes : list of int
            Size of each dense layer after the input

        dense_activation : str
            Activation functin for dense layers after input

        dense_dropout : float
            Fraction of values to randomly set to 0 during training

        dense_batch_normalization : bool
            Use Batch Normalization after hidden layers

        transform : fn, optional
            Function to transform elements of numeric input/output
        """
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.dense_batch_normalization = dense_batch_normalization
        self.transform = transform

    def build(self):
        raise NotImplementedError("Numeric cannot be directly instantiated")

    def encode(self, x):
        if self.transform:
            return self.transform(x)
        return x

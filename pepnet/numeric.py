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

class Numeric(object):
    """
    Base class for numeric input and output
    """
    def __init__(
            self,
            name,
            dim,
            dtype="float32",
            hidden_layer_sizes=[],
            hidden_activation="relu",
            hidden_dropout=0,
            batch_normalization=False,
            transform=None):
        """
        Parameters
        ----------
        name : str
            Name of input sequence

        dim : int
            Number of input dimensions

        dtype : str
            Most common option is "float32" but might also be "int32"

        hidden_layer_sizes : list of int
            Size of each dense layer after the input

        hidden_activation : str
            Activation functin for dense layers after input

        hidden_dropout : float
            Fraction of values to randomly set to 0 during training

        batch_normalization : bool
            Use Batch Normalization after hidden layers

        transform : fn, optional
            Function to transform elements of numeric input/output
        """
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_normalization = batch_normalization
        self.transform = transform

    def build(self):
        raise NotImplementedError("Numeric cannot be directly instantiated")

    def encode(self, x):
        if self.transform:
            return self.transform(x)
        return x

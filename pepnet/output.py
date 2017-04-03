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

from .helpers import dense_layers, dense
from .numeric import Numeric

class Output(Numeric):
    """
    Numeric output for neural network.
    """
    def __init__(
            self,
            dim=1,
            activation="linear",
            name=None,
            loss="mse",
            hidden_layer_sizes=[],
            hidden_activation="relu",
            hidden_dropout=0,
            batch_normalization=False,
            transform=None,
            inverse_transform=None):
        Numeric.__init__(
            self,
            name=name,
            dim=dim,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_activation=hidden_activation,
            hidden_dropout=hidden_dropout,
            batch_normalization=batch_normalization,
            transform=transform)
        self.activation = activation
        self.loss = loss
        self.inverse_transform = inverse_transform

    def build(self, value):
        hidden = dense_layers(
            value,
            layer_sizes=self.hidden_layer_sizes,
            activation=self.hidden_activation,
            dropout=self.hidden_dropout,
            batch_normalization=self.batch_normalization)
        output = dense(
            hidden,
            dim=self.dim,
            activation=self.activation,
            name=self.name)
        return output

    def decode(self, x):
        if self.inverse_transform:
            return self.inverse_transform(x)
        return x

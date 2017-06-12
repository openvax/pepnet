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

from .nn_helpers import dense_layers, make_numeric_input
from .numeric import Numeric

class NumericInput(Numeric):
    """
    Input which expects fixed length vector, takes same arguments as
    NumericOutput (defined in base class Numeric).
    """
    def build(self):
        input_object = make_numeric_input(
            name=self.name, dim=self.dim, dtype=self.dtype)
        hidden = dense_layers(
            value=input_object,
            layer_sizes=self.dense_layer_sizes,
            activation=self.dense_activation,
            dropout=self.dense_dropout,
            batch_normalization=self.dense_batch_normalization)
        return input_object, hidden

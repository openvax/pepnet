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

class DiscreteInput(Serializable):
    """
    Base class for numeric input and output
    """
    def __init__(
            self,
            choices,
            dim=1,
            name=None,
            representation="embedding",
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

        dim : int
            Number of input dimensions

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
        if representation not in {"embedding", "onehot"}:
            raise ValueError(
                ("Invalid value '%s' for representation, "
                    "must be 'embedding' or 'onehot'") % representation)

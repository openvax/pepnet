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

from keras.layers import Layer
import keras.backend as K

class DropMask(Layer):
    """
    Sometimes we know that a mask is always going to contain 1s (and never 0s)
    due to e.g. slicing the beginning of a sequence with a known min length.
    In that case it can be useful to drop the sequence mask and feed the
    activations to a layer which does not support masking (e.g. Dense).
    """
    def __init__(self, **kwargs):
        super(DropMask, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        return K.identity(x)

    def compute_mask(self, x, mask=None):
        return None

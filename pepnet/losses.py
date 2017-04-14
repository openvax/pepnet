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


import keras.backend as K
import theano.tensor as T

def positive_only_mse(y_true, y_pred):
    """
    Mean-squared error loss function that ignores negative values of y_pred.
    Using this as a stop-gap until I can figure out how to avoid the mess
    of explicitly passing an output mask as an Input to a keras model.
    """
    diff = y_pred - y_true
    squared = K.square(diff)
    mask = y_pred >= 0
    squared *= K.cast(mask, "float32")
    return K.mean(squared, axis=-1)

def masked_mse(y_true, y_pred):
    diff = y_pred - y_true
    squared = K.square(diff)
    return K.mean(
        K.switch(T.isnan(y_true), 0.0, squared),
        axis=-1)

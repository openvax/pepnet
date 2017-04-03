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

import numpy as np
from keras.models import Model

from .numeric_input import NumericInput
from .sequence_input import SequenceInput
from .output import Output
from .helpers import merge, dense_layers


class Predictor(object):
    def __init__(
            self,
            inputs,
            outputs,
            merge_mode="concat",
            hidden_layer_sizes=[],
            hidden_activation="relu",
            hidden_dropout=0.25,
            batch_normalization=False,
            optimizer="rmsprop"):

        if isinstance(inputs, (NumericInput, SequenceInput)):
            inputs = [inputs]

        if isinstance(outputs, (Output,)):
            outputs = [outputs]

        self.inputs = {i.name: i for i in inputs}
        self.input_order = [i.name for i in inputs]

        if len(outputs) > 1 and any(not o.name for o in outputs):
            raise ValueError("All outputs must have names")
        self.outputs = {o.name: o for o in outputs}
        self.output_order = [o.name for o in outputs]

        self.merge_mode = merge_mode
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_normalization = batch_normalization
        self.optimizer = optimizer
        self.model = self._build_and_compile()

    @property
    def use_input_dict(self):
        if self.num_inputs == 1:
            input_name = self.input_order[0]
            # if our single input doesn't have a name then don't try to
            # pass a dict to Keras
            return (input_name is not None) and (len(input_name) > 0)
        else:
            if any(
                    (name is None or len(name) == 0)
                    for name in self.input_order):
                raise ValueError("Predictor must have names for all %d inputs" % (
                    self.num_inputs))
            return True

    @property
    def use_output_dict(self):
        if self.num_outputs == 1:
            output_name = self.output_order[0]
            # if our single input doesn't have a name then don't try to
            # pass a dict to Keras
            return (output_name is not None) and (len(output_name) > 0)
        else:
            if any(
                    (name is None or len(name) == 0)
                    for name in self.output_order):
                raise ValueError(
                        "Predictor must have names for all %d outputs" % (
                            self.num_outputs))
            return True

    def _get_single_output(self):
        """
        When use_output_dict is False then we know that there's only one
        output and we can use it without knowing its name.
        """
        outputs = list(self.outputs.values())
        if len(outputs) == 0:
            raise ValueError("Expected at least one output")
        elif len(outputs) > 1:
            raise ValueError("Expected only one output but got %d" % (
                len(outputs)))
        return outputs[0]

    def _build(self):
        input_dict = {}
        subgraphs = []
        for (input_name, input_descriptor) in self.inputs.items():
            input_obj, subgraph = input_descriptor.build()
            input_dict[input_name] = input_obj
            subgraphs.append(subgraph)
        if len(subgraphs) == 0:
            raise ValueError("At least one input required")
        else:
            combined = merge(subgraphs, self.merge_mode)

        hidden = dense_layers(
            combined,
            layer_sizes=self.hidden_layer_sizes,
            activation=self.hidden_activation,
            dropout=self.hidden_dropout,
            batch_normalization=self.batch_normalization)

        output_dict = {}
        for output_name, output_descriptor in self.outputs.items():
            output_graph = output_descriptor.build(hidden)
            output_dict[output_name] = output_graph

        return Model(
            inputs=[input_dict[name] for name in self.input_order],
            outputs=[output_dict[name] for name in self.output_order])

    def _compile(self, model):
        if self.use_output_dict:
            loss = {
                name: self.outputs[name].loss for name in self.output_order
            }
        else:
            loss = self._get_single_output().loss
        model.compile(loss=loss, optimizer=self.optimizer)


    def _build_and_compile(self):
        model = self._build()
        self._compile(model)
        return model

    @property
    def num_inputs(self):
        return len(self.input_order)

    @property
    def num_outputs(self):
        return len(self.output_order)

    def _prepare_inputs(self, inputs):
        """
        Returns dictionary of input name -> input value if use_input_dict is
        True else, returns just encoded representation of single input.
        """
        if isinstance(inputs, (list, np.ndarray)):
            if self.num_inputs != 1:
                raise ValueError("Expected %d inputs but got 1" % self.num_inputs)
            inputs = {self.input_order[0]: inputs}
        elif not isinstance(inputs, dict):
            raise TypeError(
                "Expected inputs to be list, array, or dict, got %s" % (
                    type(inputs)))
        encoded_inputs = {
            name: self.inputs[name].encode(inputs[name])
            for name in self.input_order
        }
        if self.use_input_dict:
            return encoded_inputs
        else:
            return list(encoded_inputs.values())[0]

    def _prepare_outputs(self, outputs, encode=False, decode=False):
        if isinstance(outputs, (list, np.ndarray)):
            if self.num_outputs != 1:
                raise ValueError("Expected %d outputs but got 1" % self.num_outputs)
            outputs = {self.output_order[0]: outputs}
        elif not isinstance(outputs, dict):
            raise ValueError("Expected outputs to list, array, or dict, got %s" % (
                type(outputs)))
        if encode:
            outputs = {
                name: self.outputs[name].encode(outputs[name])
                for name in self.output_order
            }
        if decode:
            outputs = {
                name: self.outputs[name].decode(outputs[name])
                for name in self.output_order
            }
        if self.use_output_dict:
            return outputs
        else:
            return list(outputs.values())[0]

    def fit(self, inputs, outputs, batch_size=32, epochs=100):
        inputs = self._prepare_inputs(inputs)
        outputs = self._prepare_outputs(outputs)
        self.model.fit(
            inputs,
            outputs,
            batch_size=batch_size,
            epochs=epochs)

    def predict(self, inputs):
        predictions = self.model.predict(self._prepare_inputs(inputs))
        return self._prepare_outputs(predictions, decode=True)

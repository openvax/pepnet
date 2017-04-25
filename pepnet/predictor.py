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

from collections import OrderedDict

from six import string_types
import numpy as np
from keras.models import Model
from keras.utils import plot_model
from serializable import Serializable

from .numeric_input import NumericInput
from .sequence_input import SequenceInput
from .output import Output
from .helpers import merge, dense_layers


class Predictor(Serializable):
    def __init__(
            self,
            inputs,
            outputs,
            merge_mode="concat",
            dense_layer_sizes=[],
            dense_activation="relu",
            dense_dropout=0.25,
            dense_batch_normalization=False,
            optimizer="rmsprop"):

        if isinstance(inputs, (NumericInput, SequenceInput)):
            inputs = [inputs]

        if isinstance(outputs, (Output,)):
            outputs = [outputs]

        self.inputs = inputs

        if len(outputs) > 1 and any(not o.name for o in outputs):
            raise ValueError("All outputs must have names")

        self.outputs = outputs
        self.merge_mode = merge_mode
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.dense_batch_normalization = dense_batch_normalization
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
    def inputs_dict(self):
        return {i.name: i for i in self.inputs}

    @property
    def input_order(self):
        return [i.name for i in self.inputs]

    @property
    def outputs_dict(self):
        return {o.name: o for o in self.outputs}

    @property
    def output_order(self):
        return [o.name for o in self.outputs]

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
        outputs = self.outputs
        if len(outputs) == 0:
            raise ValueError("Expected at least one output")
        elif len(outputs) > 1:
            raise ValueError("Expected only one output but got %d" % (
                len(outputs)))
        return outputs[0]

    def _build(self):
        input_dict = {}
        subgraphs_dict = OrderedDict()
        for (input_name, input_descriptor) in self.inputs_dict.items():
            input_obj, subgraph = input_descriptor.build()
            input_dict[input_name] = input_obj
            subgraphs_dict[input_name] = subgraph

        if len(subgraphs_dict) == 0:
            raise ValueError("At least one input required")
        elif isinstance(self.merge_mode, string_types):
            merge_dict = {tuple(subgraphs_dict.keys()): self.merge_mode}
        elif not isinstance(self.merge_mode, dict):
            raise TypeError("Expected 'merge' to be str or dict but got %s : %s " % (
                self.merge_mode, type(self.merge_mode)))
        else:
            merge_dict = self.merge_mode

        combined_tensors = []
        for (input_names, merge_mode) in merge_dict.items():
            if isinstance(input_names, string_types):
                input_names = (input_names,)
            combined_tensors.append(
                merge([subgraphs_dict[name] for name in input_names],
                        merge_mode))
        # concatenate final tensor in case we're combining more than one
        # group
        combined = merge(combined_tensors, "concat")

        hidden = dense_layers(
            combined,
            layer_sizes=self.dense_layer_sizes,
            activation=self.dense_activation,
            dropout=self.dense_dropout,
            batch_normalization=self.dense_batch_normalization)

        output_dict = {}
        for output_name, output_descriptor in self.outputs_dict.items():
            output_graph = output_descriptor.build(hidden)
            output_dict[output_name] = output_graph

        return Model(
            inputs=[input_dict[name] for name in self.input_order],
            outputs=[output_dict[name] for name in self.output_order])

    def _compile(self, model):
        if self.use_output_dict:
            loss = {
                output.name: output.loss_fn for output in self.outputs
            }
        else:
            loss = self._get_single_output().loss_fn
        model.compile(loss=loss, optimizer=self.optimizer)

    def _build_and_compile(self):
        if self.num_inputs == 0:
            raise ValueError("Predictor must have at least one output")
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
            name: i.encode(inputs[name])
            for name, i in self.inputs_dict.items()
        }
        lengths = {name: len(x) for (name, x) in encoded_inputs.items()}
        if any(l != list(lengths.values())[0] for l in lengths.values()):
            raise ValueError("All inputs must be of the same length, given %s" % (
                lengths,))
        if self.use_input_dict:
            return encoded_inputs
        else:
            return list(encoded_inputs.values())[0]

    def _prepare_outputs(self, outputs, encode=False, decode=False):
        if isinstance(outputs, list):
            outputs = np.array(outputs).squeeze().T

        if isinstance(outputs, np.ndarray):
            if outputs.ndim == 1:
                outputs = np.expand_dims(outputs, 1)

            n_given_outputs = outputs.shape[1]

            if self.num_outputs != n_given_outputs:
                raise ValueError("Expected %d outputs but got %d" % (
                    self.num_outputs,
                    n_given_outputs))

            outputs = {
                output_name: outputs[:, i]
                for i, output_name
                in enumerate(self.output_order)
            }
        elif not isinstance(outputs, dict):
            raise ValueError("Expected outputs to list, array, or dict, got %s" % (
                type(outputs)))
        if encode:
            outputs = {
                name: output.encode(outputs[name])
                for name, output in self.outputs_dict.items()
            }
        if decode:
            outputs = {
                name: output.decode(outputs[name])
                for name, output in self.outputs_dict.items()
            }
        lengths = {name: len(x) for (name, x) in outputs.items()}
        if any(l != list(lengths.values())[0] for l in lengths.values()):
            raise ValueError(
                "All outputs must be of the same length, given %s" % (lengths,))
        if self.use_output_dict:
            return outputs
        else:
            return list(outputs.values())[0]

    def fit(self,
            inputs,
            outputs,
            batch_size=32,
            epochs=100,
            sample_weight=None,
            class_weight=None):

        inputs = self._prepare_inputs(inputs)
        outputs = self._prepare_outputs(outputs, encode=True)

        if sample_weight is not None:
            if self.use_input_dict and len(self.outputs) > 1:
                if isinstance(sample_weight, np.ndarray):
                    sample_weight = {
                        o.name: sample_weight
                        for o in self.outputs
                    }

        self.model.fit(
            inputs,
            outputs,
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=sample_weight,
            class_weight=class_weight,
            shuffle=True)

    def predict_scores(self, inputs):
        return self._prepare_outputs(
            self.model.predict(self._prepare_inputs(inputs)),
            decode=False)

    def predict(self, inputs):
        return self._prepare_outputs(
            self.model.predict(self._prepare_inputs(inputs)),
            decode=True)

    def save_diagram(self, filename="model.png"):
        plot_model(self.model, to_file=filename)

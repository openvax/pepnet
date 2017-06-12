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
import ujson

from .numeric_input import NumericInput
from .sequence_input import SequenceInput
from .output import Output
from .nn_helpers import merge, dense_layers, tensor_shape


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
            optimizer="rmsprop",
            training_metrics=[]):

        if isinstance(inputs, (NumericInput, SequenceInput)):
            inputs = [inputs]
        elif isinstance(inputs, dict):
            inputs_dict = inputs
            inputs = []
            for (name, i) in sorted(inputs_dict.items()):
                if i.name is None:
                    i.name = name
                elif i.name != name:
                    raise ValueError("Input named '%s' given key '%s'" % (i.name, name))
                inputs.append(i)

        if isinstance(outputs, (Output,)):
            outputs = [outputs]
        elif isinstance(outputs, dict):
            outputs_dict = outputs
            outputs = []
            for (name, o) in sorted(outputs_dict.items()):
                if o.name is None:
                    o.name = name
                elif o.name != name:
                    raise ValueError("Output named '%s' given key '%s'" % (o.name, name))
                outputs.append(o)

        if len(outputs) > 1 and any(not o.name for o in outputs):
            raise ValueError("Predictors with multiple outputs must have names for each output")
        if len(outputs) > len({o.name for o in outputs}):
            raise ValueError("All outputs must have distinct names")

        self.inputs = inputs
        self.outputs = outputs
        self.merge_mode = merge_mode
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.dense_batch_normalization = dense_batch_normalization
        self.optimizer = optimizer
        self.training_metrics = training_metrics
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
    def output_names(self):
        return self.output_order

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

        inputs = [input_dict[name] for name in self.input_order]
        outputs = [output_dict[name] for name in self.output_order]
        return Model(inputs=inputs, outputs=outputs)

    def _compile(self, model):
        if self.use_output_dict:
            loss = {
                output.name: output.loss_fn for output in self.outputs
            }
        else:
            loss = self._get_single_output().loss_fn
        model.compile(loss=loss, optimizer=self.optimizer, metrics=self.training_metrics)

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

    ############################################################################
    #
    # Prediction
    #
    ############################################################################

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
        """
        Returns a dictionary from output name to array of output values.
        """
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

    def predict_scores(self, inputs):
        return self._prepare_outputs(
            self.model.predict(self._prepare_inputs(inputs)),
            decode=False)

    def predict(self, inputs):
        return self._prepare_outputs(
            self.model.predict(self._prepare_inputs(inputs)),
            decode=True)

    ############################################################################
    #
    # Weight estimation
    #
    ############################################################################

    def _prepare_sample_weights(self, sample_weight):
        if sample_weight is not None:
            if self.use_input_dict and len(self.outputs) > 1:
                if isinstance(sample_weight, np.ndarray):
                    sample_weight = {
                        o.name: sample_weight
                        for o in self.outputs
                    }
        return sample_weight

    def _prepare_data_tuple(self, data_tuple):
        """
        Data generators return either (input, output) or
        (input, output, weights) tuples. This function transforms
        these elements for use with Keras.
        """
        if len(data_tuple) == 2:
            inputs, outputs = data_tuple
            weights = None
        else:
            assert len(data_tuple) == 3, \
                "Dataset expected to be (X, Y, weights), got %d elements" % (
                    len(data_tuple),)
            inputs, outputs, weights = data_tuple
        inputs = self._prepare_inputs(inputs)
        outputs = self._prepare_outputs(outputs)
        weights = self._prepare_sample_weights(weights)
        return inputs, outputs, weights

    def _wrap_data_generator(self, generator):
        for data_tuple in generator:
            yield self._prepare_data_tuple(data_tuple)

    def fit(self,
            inputs,
            outputs,
            batch_size=32,
            epochs=100,
            sample_weight=None,
            class_weight=None,
            validation_data=None,
            shuffle=True,
            callbacks=[]):
        inputs = self._prepare_inputs(inputs)
        outputs = self._prepare_outputs(outputs, encode=True)
        sample_weight = self._prepare_sample_weights(sample_weight)

        if validation_data is not None:
            validation_data = self._prepare_data_tuple(validation_data)

        return self.model.fit(
            inputs,
            outputs,
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=sample_weight,
            class_weight=class_weight,
            shuffle=shuffle,
            validation_data=validation_data,
            callbacks=callbacks)

    def fit_generator(
            self,
            generator,
            steps_per_epoch,
            **kwargs):
        """
        Expects a generator which returns tuples of
            (inputs, outputs)
            -or-
            (inputs, outputs, sample_weights)
        which are then transformed appropriately before being
        passed on to the fit_generator method of the underlying
        Keras model.
        """
        return self.model.fit_generator(
            generator=self._wrap_data_generator(generator),
            steps_per_epoch=steps_per_epoch,
            **kwargs)

    ############################################################################
    #
    # Model visualization
    #
    ############################################################################

    def save_diagram(self, filename="model.png"):
        plot_model(self.model, to_file=filename)

    ############################################################################
    #
    # Serialization methods and related helpers
    #
    ############################################################################

    def _input_to_repr(self, input_obj):
        """
        Return a serializable representation of an input.
        """
        return (input_obj.__class__.__name__, input_obj.to_dict())

    @classmethod
    def _input_from_repr(cls, input_repr):
        """
        Create an input from a flattened representation
        """
        name, config = input_repr
        if name == "SequenceInput":
            return SequenceInput.from_dict(config)
        elif name == "NumericInput":
            return NumericInput.from_dict(config)
        else:
            raise ValueError("Invalid input class: %s" % (name,))

    def _output_to_repr(self, output_obj):
        return (output_obj.__class__.__name__, output_obj.to_dict())

    @classmethod
    def _output_from_repr(self, output_repr):
        name, config = output_repr
        if name == "Output":
            return Output.from_dict(config)
        else:
            raise ValueError("Invalid output name: %s" % (name,))

    def get_weights(self):
        return [w.get_value().squeeze() for w in self.model.weights]

    def set_weights(self, weights):
        if len(self.model.weights) != len(weights):
            raise ValueError("Expected %d weight arrays but got %d" % (
                len(self.model.weights),
                len(weights)))
        for w_tensor, w_values in zip(self.model.weights, weights):
            shape = tensor_shape(w_tensor)
            w_compatible = w_values.reshape(shape)
            w_compatible = w_compatible.astype(w_tensor.dtype)
            w_tensor.set_value(w_compatible)

    def to_dict(self):
        return {
            "inputs": [self._input_to_repr(i) for i in self.inputs],
            "outputs": [self._output_to_repr(o) for o in self.outputs],
            "merge_mode": self.merge_mode,
            "dense_layer_sizes": self.dense_layer_sizes,
            "dense_activation": self.dense_activation,
            "dense_dropout": self.dense_dropout,
            "dense_batch_normalization": self.dense_batch_normalization,
            "optimizer": self.optimizer,
            "training_metrics": self.training_metrics,
            "model_weights": [w.tolist() for w in self.get_weights()]
        }

    @classmethod
    def from_dict(self, config_dict):
        model_weights_as_lists = config_dict.pop("model_weights")
        model_weights = [np.array(values) for values in model_weights_as_lists]
        input_reprs = config_dict.pop("inputs")
        output_reprs = config_dict.pop("outputs")
        inputs = [self._input_from_repr(i) for i in input_reprs]
        outputs = [self._output_from_repr(o) for o in output_reprs]
        predictor = Predictor(inputs=inputs, outputs=outputs, **config_dict)
        predictor.set_weights(model_weights)
        return predictor

    def to_json(self):
        return ujson.dumps(self.to_dict())

    def to_json_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_string):
        config_dict = ujson.loads(json_string)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r") as f:
            s = f.read()
            if len(s) == 0:
                raise ValueError("File '%s' is empty" % filename)
            return cls.from_json(s)

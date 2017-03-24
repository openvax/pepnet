from .helpers import merge, dense_layers
from keras import Model

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
        self.inputs = inputs
        self.outputs = outputs
        self.merge_mode = merge_mode
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_normalization = batch_normalization
        self.optimizer = optimizer

        self.model = self._build_and_compile()

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

        return Model(inputs=input_dict, outputs=output_dict)

    def _compile(self, model):
        loss_dict = {output.name: output.loss for output in self.outputs}
        model.compile(loss=loss_dict, optimizer=self.optimizer)

    def _build_and_compile(self):
        model = self.build()
        self._compile(model)
        return model

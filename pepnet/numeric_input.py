from .helpers import dense_layers, make_numeric_input

class NumericInput(object):
    def __init__(
            self,
            name,
            dim,
            dtype="float32",
            hidden_layer_sizes=[],
            hidden_activation="relu",
            hidden_dropout=0,
            batch_normalization=False):
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
        """
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_normalization = batch_normalization

    def build(self):
        input_object = make_numeric_input(
            name=self.name, dim=self.dim, dtype=self.dtype)
        hidden = dense_layers(
            input_object,
            layer_sizes=self.hidden_layer_sizes,
            activation=self.hidden_activation,
            dropout=self.hidden_dropout,
            batch_normalization=self.batch_normalization)
        return input_object, hidden

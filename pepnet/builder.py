
from keras.layers import (Dense, Embedding, Input, Dropout)
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import Concatenate
from keras.layers.pooling import (
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D
)
from keras.models import Model

class Sequence(object):
    def __init__(
            self,
            name,
            length,
            n_symbols,
            encoding="index",
            input_object=None,
            value=None):
        self.name = name
        self.length = length
        self.n_symbols = n_symbols
        self.encoding = encoding

        if input_object is None:
            assert encoding in {"onehot", "index"}
            if encoding == "index":
                input_object = Input(
                    shape=(length,),
                    dtype="int32",
                    name=name)
            else:
                input_object = Input(
                    shape=(length * n_symbols,),
                    name=name,
                    dtype="float32")
        self.input_object = input_object
        if value is None:
            value = input_object
        self.value = value

    def copy_with(self, value):
        """
        Create copy of this object with updated value
        """
        return self.__class__(
            name=self.name,
            length=self.length,
            n_symbols=self.n_symbols,
            encoding=self.encoding,
            input_object=self.input_object,
            value=value)

    def embedding(self, output_dim, dropout=0, initial_weights=None):
        if initial_weights:
            n_rows, n_cols = initial_weights.shape
            if n_rows != self.n_symbols or n_cols != output_dim:
                raise ValueError(
                    "Wrong shape for embedding: expected (%d, %d) but got "
                    "(%d, %d)" % (
                        self.n_symbols, output_dim,
                        n_rows, n_cols))
            embedding_layer = Embedding(
                input_dim=n_symbols,
                output_dim=output_dim,
                mask_zero=False,
                weights=[initial_weights],
                name="%s_embedding" % self.name)
        else:
            embedding_layer = Embedding(
                input_dim=n_symbols,
                output_dim=output_dim,
                mask_zero=False,
                name="%s_embedding" % self.name)

        value = embedding_layer(self.value)

        if embedding_dropout:
            value = SpatialDropout1D(embedding_dropout)(value)

        return self.copy_with(value)

    def conv(self, filter_sizes, output_dim, dropout=0.1):
        """
        Parameters
        ----------
        filter_sizes : int or list of int
            Widths of convolutional filters

        output_dim : int
            Number of filters per width

        dropout : float
            Dropout after convolutional
        """
        if isinstance(filter_sizes, int):
            filter_sizes = [filter_sizes]

        convolved_list = []
        for size in filter_sizes:
            conv_layer = Convolution1D(
                filters=output_dim,
                kernel_size=size,
                padding="same",
                name="%s_conv_layer_size_%d" % (self.name, size))
            convolved = conv_layer(self.value)
            if dropout:
                # random drop some of the convolutional activations
                convolved = Dropout(dropout)(convolved)

            convolved_list.append(convolved)
        if len(convolved_list) > 1:
            convolved = Concatenate()(convolved_list)
        else:
            convolved = convolved_list[0]
        return self.copy_with(convolved)

    def max_pool(self, size=3, stride=2):


class NetworkBuilder(object):

    def __init__(self):
        self.input_names = []
        # dictionary from input names to keras Input objects
        self.inputs = {}
        # dictionary from input names to transformed inputs
        self.inputs_after_transformation = {}
        self.output_names = []
        # dictionary from output name to tensor
        self.outputs = {}
        # dictionary from output name to loss
        self.losses = {}

    def add_sequence_input(
            self,
            name,
            length,
            n_symbols=21,
            embedding_dim=None,
            embedding_dropout=0,
            embedding_initial_weights=None,
            n_conv_layers=0,
            conv_filter_sizes=[3],
            conv_dropout=0.1,
            conv_pool_size=3,
            conv_pool_stride=2):
        if name in self.inputs:
            raise ValueError("Input '%s' already added to network" % (name,))
        self.input_names.append(name)

        if not dtype:dtype = "int32" if embedding_dim else "float32"

        if embedding_dim:
            # if we're given an embedding dim then the input sequence
            # has to be index encoded
            input_value = Input(shape=(length,), "int32", name=name)
            self.inputs[name] = input_value

            embedding_layer = Embedding(
                input_dim=n_symbols,
                output_dim=embedding_output_dim,
                mask_zero=False,
                name="%s-embedding" % name)
            x = embedding_layer(input_value)
            if embedding_dropout:
                x = SpatialDropout1D(embedding_dropout)(x)
        else:
            # without an embedding we just expect a one-hot encoding
            input_value = Input(
                shape=(length * n_symbols,), "float32", name=name)


        if initial_embedding_weights:
            n_rows, n_cols = initial_embedding_weights.shape
            if n_rows != embedding_input_dim or n_cols != embedding_output_dim:
                raise ValueError(
                    "Wrong shape for embedding: expected (%d, %d) but got "
                    "(%d, %d)" % (
                        embedding_input_dim, embedding_output_dim,
                        n_rows, n_cols))
            model.add(Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=input_size,
                weights=[initial_embedding_weights]))

def make_variable_length_embedding_convolutional_model(
        n_symbols=21,
        max_peptide_length=30,
        embedding_output_dim=32,
        n_filters_per_size=32,
        filter_sizes=[3, 5, 9],
        n_conv_layers=1,
        pool_size=3,
        pool_stride=2,
        dropout=0.25,
        conv_dropout=0.1,
        hidden_layer_sizes=[20],
        hidden_activation="relu",
        optimizer="rmsprop",
        loss="mse",
        n_output=1,
        output_activation="sigmoid",
        compile_kwargs={}):
    """
    Make a model with a single peptide input which is passed through a
    learned amino acid embedding and then passed through alternating
    layers of 1d convolution and max pooling. At the end all of the filters
    are put through global max pooling, passed through one or more hidden layers
    and then transformed into an output.
    """
    # we're not masking the '-' character (expected to be at index 0) since
    # 1d convolutions don't yet work with masking, so we're just throwing
    # the characters after the end of the peptide into the embedding and
    # hoping for the best
    if isinstance(filter_sizes, int):
        filter_sizes = [filter_sizes]

    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = [hidden_layer_sizes]

    # peptide
    peptide = Input(shape=(max_peptide_length,), dtype="int32", name="peptide")

    peptide_embedding = Embedding(
        input_dim=n_symbols + 1,
        output_dim=embedding_output_dim,
        mask_zero=False,
        name="peptide-embedding")

    x = peptide_embedding(peptide)

    for i in range(n_conv_layers):
        convolved_list = []
        for size in filter_sizes:
            conv_layer = Convolution1D(
                filters=n_filters_per_size,
                kernel_size=size,
                padding="same",
                name="conv_layer_%d_size_%d" % (i + 1, size))
            convolved = conv_layer(x)
            if conv_dropout:
                # random drop some of the convolutional activations
                convolved = Dropout(conv_dropout)(convolved)
            # add max pooling for all layers before the last
            if i + 1 != n_conv_layers:
                convolved = MaxPooling1D(
                        pool_size=pool_size,
                        strides=pool_stride,
                        name="maxpool_%d_for_filter_size_%d" % (
                            i + 1,
                            size))(convolved)
            convolved_list.append(convolved)
        if len(convolved_list) > 1:
            x = Concatenate()(convolved_list)
        else:
            x = convolved_list[0]

    global_max = GlobalMaxPooling1D()(x)
    global_avg = GlobalAveragePooling1D()(x)

    x = Concatenate()([global_max, global_avg])
    for i, n_hidden in enumerate(hidden_layer_sizes):
        x = Dense(
            n_hidden,
            activation=hidden_activation,
            name="hidden_%d" % (i + 1,))(x)
        x = Dropout(dropout)(x)

    output = Dense(n_output, activation=output_activation, name="output")(x)
    model = Model(
        inputs=peptide,
        outputs=output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        **compile_kwargs)
    return model

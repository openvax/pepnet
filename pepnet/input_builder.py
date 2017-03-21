
from keras.layers import (Embedding, Input, Dropout, SpatialDropout1D)
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import Concatenate
from keras.layers.pooling import (
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D
)

class InputBuilder(object):
    """
    Class used to build up one input path through a neural network before
    dense hidden layers.
    """
    def __init__(
            self,
            name,
            length,
            n_symbols=21,
            encoding="index",
            input_object=None,
            value=None):
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

    def apply(self, layer):
        return self.copy_with(layer(self.value))

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
                input_dim=self.n_symbols,
                output_dim=output_dim,
                mask_zero=False,
                weights=[initial_weights],
                name="%s_embedding" % self.name)
        else:
            embedding_layer = Embedding(
                input_dim=self.n_symbols,
                output_dim=output_dim,
                mask_zero=False,
                name="%s_embedding" % self.name)

        value = embedding_layer(self.value)

        if dropout:
            value = SpatialDropout1D(dropout)(value)
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
                padding="same")
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

    def local_max_pooling(self, size=3, stride=2):
        return self.apply(MaxPooling1D(pool_size=size, strides=stride))

    def global_max_pooling(self):
        return self.apply(GlobalMaxPooling1D())

    def global_mean_pooling(self):
        return self.apply(GlobalAveragePooling1D())

    def global_max_and_mean_pooling(self):
        max_pooled = GlobalMaxPooling1D()(self.value)
        mean_pooled = GlobalAveragePooling1D(self.value)
        concat = Concatenate([max_pooled, mean_pooled])
        return self.copy_with(concat)

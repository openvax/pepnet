from keras.layers import (Dense, Embedding, Input, Dropout)
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import Concatenate
from keras.layers.pooling import (
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D
)
from keras.models import Model


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

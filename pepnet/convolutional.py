from keras.layers import (Dense, Embedding, Input, Dropout, merge)
from keras.layers.convolutional import Conv1D
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
        embedding_dropout=0,
        n_hidden=20,
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
    are put through global max pooling, passed through one hidden layer
    and then transformed into an output.
    """
    # we're not masking the '-' character (expected to be at index 0) since
    # 1d convolutions don't yet work with masking, so we're just throwing
    # the characters after the end of the peptide into the embedding and
    # hoping for the best

    # peptide
    peptide = Input(shape=(max_peptide_length,), dtype="int32", name="peptide")

    peptide_embedding = Embedding(
        input_dim=n_symbols + 1,
        output_dim=embedding_output_dim,
        mask_zero=False,
        name="peptide-embedding",
        dropout=embedding_dropout)

    x = peptide_embedding(peptide)

    for i in range(n_conv_layers):
        convolved = [
            Conv1D(
                filters=n_filters_per_size,
                kernel_size=size,
                padding="same")(x)
            for size in filter_sizes
        ]
        x = merge(convolved, mode="concat")
        if conv_dropout:
            # random drop some of the convolutional activations
            x = Dropout(conv_dropout)(x)
        # add max pooling for all layers before the last
        if i + 1 != n_conv_layers:
            x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)

    global_max = GlobalMaxPooling1D()(x)
    global_avg = GlobalAveragePooling1D()(x)
    global_concat = merge([global_max, global_avg], mode="concat")

    hidden = Dense(n_hidden, activation=hidden_activation, name="hidden")(global_concat)
    hidden = Dropout(dropout)(hidden)
    output = Dense(n_output, activation=output_activation, name="output")(hidden)
    model = Model(
        input=peptide,
        output=output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        **compile_kwargs)
    return model

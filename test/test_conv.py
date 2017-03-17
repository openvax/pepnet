from pepnet.convolutional import (
    make_variable_length_embedding_convolutional_model,)

import numpy as np

def test_embedding_conv_1_layer():
    model = make_variable_length_embedding_convolutional_model(
        max_peptide_length=3,
        n_conv_layers=1)
    X = np.array([[2, 1, 0], [4, 6, 2]])
    Y = np.array([True, False])
    model.fit(X, Y)

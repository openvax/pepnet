from pepnet import Predictor, SequenceInput, Output
import numpy as np

def test_embedding_conv_1_layer():
    model = Predictor(
        inputs=SequenceInput(
            length=3, variable_length=False, conv_filter_sizes={2: 3}),
        outputs=Output(1, activation="sigmoid"))
    X = ["SAY", "FFQ"]
    Y = np.array([True, False])
    model.fit(X, Y)

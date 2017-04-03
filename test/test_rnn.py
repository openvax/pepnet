from pepnet import Predictor, SequenceInput, Output
from nose.tools import eq_

def test_basic_rnn():
    pred = Predictor(
        inputs=SequenceInput(
            name="x",
            length=4,
            variable_length=True,
            encoding="index",
            rnn_layer_sizes=[20],
            rnn_type="lstm",
            rnn_bidirectional=True),
        outputs=Output(dim=1, activation="sigmoid", name="y"))
    x = ["SF", "Y", "AALL"]
    y = pred.predict({"x": x})["y"]
    eq_(len(x), len(y))
    found_rnn_layer = any(
        "bidirectional" in layer.name for layer in pred.model.layers)
    assert found_rnn_layer

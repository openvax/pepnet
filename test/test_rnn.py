from pepdata import Predictor, SequenceInput, Output
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
            rnn_dropout=0.25,
            rnn_bidirectional=True),
        outputs=Output(dim=1, name="y"))
    x = ["SF", "Y", "AALL"]
    y = pred.predict(x=x)["y"]
    assert eq_(len(x), len(y))
    found_rnn_layer = False
    for layer in pred.model.layers:
        print(layer)
        if "LSTM" in layer.name:
            found_rnn_layer = True
    assert found_rnn_layer

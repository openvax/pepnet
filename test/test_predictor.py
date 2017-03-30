from pepnet import Predictor, SequenceInput, NumericInput, Output
from numpy.random import randn
from numpy import log, exp
from nose.tools import eq_


def test_simple_numeric_predictor():
    predictor = Predictor(
        inputs=[NumericInput(dim=30)],
        outputs=[Output(dim=1, activation="sigmoid")],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict(randn(10, 30))
    eq_(len(y), 10)

def test_simple_numeric_predictor_named():
    predictor = Predictor(
        inputs=[NumericInput(name="x", dim=30)],
        outputs=[Output(dim=1, name="y", activation="sigmoid")],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({"x": randn(10, 30)})["y"]
    eq_(len(y), 10)


def test_simple_sequence_predictor():
    predictor = Predictor(
        inputs=[SequenceInput(length=4, variable_length=True)],
        outputs=[Output(dim=1, activation="sigmoid")],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict(["SFY-"] * 10)
    eq_(len(y), 10)

def test_simple_sequence_predictor_named():
    predictor = Predictor(
        inputs=[SequenceInput(length=4, name="x", variable_length=True)],
        outputs=[Output(dim=1, activation="sigmoid", name="y")],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({"x": ["SFY-"] * 10})["y"]
    eq_(len(y), 10)

def test_two_input_predictor():
    predictor = Predictor(
        inputs=[
            SequenceInput(length=4, name="x1", variable_length=True),
            NumericInput(dim=30, name="x2")],
        outputs=[Output(name="y", dim=1, activation="sigmoid")],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({
        "x1": ["SFY-"] * 10,
        "x2": randn(10, 30)})["y"]
    eq_(len(y), 10)

def test_predictor_output_transform():
    predictor = Predictor(
        inputs=[NumericInput(dim=30, name="x")],
        outputs=[
            Output(
                name="y",
                dim=1,
                activation="sigmoid",
                transform=log,
                inverse_transform=exp)],
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({"x": randn(10, 30)})["y"]
    eq_(len(y), 10)
    # make sure transformed outputs are within given bounds
    assert exp(0.0) <= y.min() <= exp(1.0)
    assert exp(0.0) <= y.max() <= exp(1.0)

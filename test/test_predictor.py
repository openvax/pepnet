from pepnet import Predictor, SequenceInput, NumericInput, Output
import numpy as np
from nose.tools import eq_


def test_simple_numeric_predictor():
    predictor = Predictor(
        inputs={"x": NumericInput(30)},
        outputs={"y": Output(1, "sigmoid")},
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({"x": np.random.randn(10, 30)})["y"]
    eq_(len(y), 10)


def test_simple_sequence_predictor():
    predictor = Predictor(
        inputs={"x": SequenceInput(4)},
        outputs={"y": Output(1, "sigmoid")},
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({"x": ["SFY-"] * 10})["y"]
    eq_(len(y), 10)


def test_two_input_predictor():
    predictor = Predictor(
        inputs={"x1": SequenceInput(4), x2: NumericInput(30)},
        outputs={"y": Output(1, "sigmoid")},
        hidden_layer_sizes=[30],
        hidden_activation="relu")
    y = predictor.predict({
        "x1": ["SFY-"] * 10,
        "x2": np.random.randn(10, 30)})["y"]
    eq_(len(y), 10)


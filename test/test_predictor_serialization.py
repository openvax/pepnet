from pepnet import Predictor, SequenceInput, Output
from nose.tools import eq_
import numpy as np

def test_predictor_json_identity():
    predictor = Predictor(
        inputs=[SequenceInput(length=2, variable_length=False, encoding="onehot")],
        outputs=[Output(dim=1, activation="sigmoid")])
    eq_(predictor, Predictor.from_json(predictor.to_json()))

def test_predictor_weights_all_ones():
    predictor = Predictor(
        inputs=[SequenceInput(length=2, variable_length=False, encoding="onehot")],
        outputs=[Output(dim=1, activation="sigmoid")])
    weights = predictor.get_weights()
    for w in weights:
        w.fill(1)
    predictor.set_weights(weights)
    predictor2 = Predictor.from_json(predictor.to_json())
    for w in predictor2.get_weights():
        assert (w == np.ones_like(w)).all(), "Expected %s to be all 1s" % (w,)

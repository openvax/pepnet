from pepnet import Predictor, SequenceInput, Output
from nose.tools import eq_

def test_predictor_json_identity():
    predictor = Predictor(
        inputs=[SequenceInput(length=2, variable_length=False, encoding="onehot")],
        outputs=[Output(dim=1, activation="sigmoid")])
    eq_(predictor, Predictor.from_json(predictor.to_json()))

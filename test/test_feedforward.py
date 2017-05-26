from pepnet import SequenceInput, Output, Predictor
from nose.tools import eq_

def test_fixed_length_hotshot():
    model = Predictor(
        inputs=SequenceInput(length=9, variable_length=False, encoding="onehot"),
        outputs=Output(1, activation="sigmoid"))
    seqs = ["A" * 9, "L" * 9]
    y = model.predict(seqs)
    eq_(len(y), 2)

def test_fixed_length_embedding_network():
    model = Predictor(
        inputs=SequenceInput(length=9, variable_length=False, encoding="index"),
        outputs=Output(1, activation="sigmoid"))
    seqs = ["A" * 9, "L" * 9]
    y = model.predict(seqs)
    eq_(len(y), 2)

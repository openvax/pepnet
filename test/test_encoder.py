from pepnet.encoder import Encoder
from nose.tools import eq_
import numpy as np

def test_encoder_index_lists():
    encoder = Encoder()
    S_idx = encoder.index_dict["S"]
    A_idx = encoder.index_dict["A"]
    index_lists = encoder.encode_index_lists(["SSS", "AAA", "SAS"])
    eq_(index_lists, [
        [S_idx, S_idx, S_idx],
        [A_idx, A_idx, A_idx],
        [S_idx, A_idx, S_idx]
    ])

def test_encoder_prepare_sequences_padding():
    encoder = Encoder()
    eq_(encoder.prepare_sequences(["SISI"], 5), ["SISI-"])

def test_encoder_prepare_sequences_start_token():
    encoder = Encoder(add_start_tokens=True)
    eq_(encoder.prepare_sequences(["SISI"], 5), ["^SISI-"])


def test_encoder_prepare_sequences_stop_token():
    encoder = Encoder(add_stop_tokens=True)
    eq_(encoder.prepare_sequences(["SISI"], 5), ["SISI$-"])


def test_encoder_index_array():
    encoder = Encoder()
    S_idx = encoder.index_dict["S"]
    A_idx = encoder.index_dict["A"]
    assert S_idx > 0
    assert A_idx > 0
    X = encoder.encode_index_array(["SSS", "AAA", "SASA"], max_peptide_length=4)
    expected = np.array([
        [S_idx, S_idx, S_idx, 0],
        [A_idx, A_idx, A_idx, 0],
        [S_idx, A_idx, S_idx, A_idx]
    ])
    assert (X == expected).all()


def test_encoder_FOFE():
    # turn off the gap character '-' used for ends of shorter sequences
    encoder = Encoder(variable_length_sequences=False)
    x = encoder.encode_FOFE(["AAA", "SSS", "SASA"])
    eq_(x.shape, (3, 20))

def test_encoder_FOFE_bidirectional():
    # turn off the gap character '-' used for ends of shorter sequences
    encoder = Encoder(variable_length_sequences=False)
    x = encoder.encode_FOFE(["AAA", "SSS", "SASA"], bidirectional=True)
    eq_(x.shape, (3, 40))

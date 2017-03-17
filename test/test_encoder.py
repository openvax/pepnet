from pepnet.encoder import Encoder
from nose.tools import eq_

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


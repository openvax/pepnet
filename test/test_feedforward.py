from pepnet.feed_forward import (
    make_fixed_length_hotshot_network,
    make_fixed_length_embedding_network
)
import numpy as np
from nose.tools import eq_

def test_fixed_length_hotshot():
    model = make_fixed_length_hotshot_network(
        peptide_length=9,
        n_symbols=20)
    x = np.array([
        [0] * (9 * 20),
        [1] * (9 * 20)
    ])
    y = model.predict(x)
    eq_(len(y), 2)

def test_fixed_length_embedding_network():
    model = make_fixed_length_embedding_network(
        peptide_length=5,
        n_symbols=20)
    x = np.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]])
    y = model.predict(x)
    eq_(len(y), 2)

from pepnet.amino_acids import amino_acids_with_ptms
from nose.tools import eq_

def test_ptm_dict_no_args():
    alphabet = amino_acids_with_ptms([])
    eq_(len(alphabet), 20)

def test_ptm_dict_two_args():
    alphabet = amino_acids_with_ptms(["c", "m"])
    eq_(len(alphabet), 22)

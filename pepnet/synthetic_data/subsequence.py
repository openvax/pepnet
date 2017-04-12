import numpy
import pandas

from ..amino_acids import amino_acids_dict
from .random_peptides import random_peptides

AMINO_ACIDS = list(amino_acids_dict)


def synthetic_peptides_by_subsequence(
        num_peptides,
        fraction_binders=0.5,
        lengths=range(8, 20),
        binding_subsequences=["A?????Q"]):
    """
    Generate a toy dataset where each peptide is a binder if and only if it
    has one of the specified subsequences.
    
    Parameters
    ----------
    num_peptides : int
        Number of rows in result
    
    fraction_binders : float
        Fraction of rows in result where "binder" col is 1
    
    lengths : dict, Series, or list
        If a dict or Series, then this should map lengths to the fraction of the
        result to have the given peptide length. If it's a list of lengths then
        all lengths are given equal weight.
    
    binding_subsequences : list of string
        Peptides with any of the given subsequences will be considered binders.
        Question marks ("?") in these sequences will be replaced by random
        amino acids.
        
    Returns
    ----------
    pandas.DataFrame, indexed by peptide sequence. The "binder" column is a
    binary indicator for whether the peptide is a binder.
    """
    if not isinstance(lengths, dict):
        lengths = dict((length, 1.0) for length in lengths)

    lengths_series = pandas.Series(lengths)
    lengths_series /= len(lengths)

    num_binders = int(round(num_peptides * fraction_binders))
    num_non_binders = num_peptides - num_binders
    print(num_binders, num_non_binders)

    peptides = []

    # Generate non-binders
    for (length, weight) in lengths_series.iteritems():
        peptides.extend(
            random_peptides(round(weight * num_non_binders), round(length)))

    for binding_core in binding_subsequences:
        # Generate binders
        lengths_binders = lengths_series.ix[
            lengths_series.index >= len(binding_core)
        ]
        normalized_lengths_binders = (
            lengths_binders /
            lengths_binders.sum() /
            len(binding_subsequences))

        for (length, weight) in normalized_lengths_binders.iteritems():
            if length >= len(binding_core):
                num_peptides_to_make = int(round(weight * num_binders))
                if length == len(binding_core):
                    start_positions = [0] * num_peptides_to_make
                else:
                    start_positions = numpy.random.choice(
                        length - len(binding_core), num_peptides_to_make)
                peptides.extend(
                    "".join([
                        random_peptides(1, length=start_position)[0],
                        binding_core,
                        random_peptides(1, length=length - len(
                            binding_core) - start_position)[0],
                    ])
                    for start_position in start_positions)

    df = pandas.DataFrame(index=set(peptides))
    df["binder"] = False
    for binding_core in binding_subsequences:
        df["binder"] = df["binder"] | df.index.str.contains(
            binding_core,
            regex=False)

    def replace_question_marks(s):
        while "?" in s:
            s = s.replace("?", numpy.random.choice(AMINO_ACIDS))
        return s

    df.index = df.index.map(replace_question_marks)
    df_shuffled = df.sample(frac=1)
    return df_shuffled

import numpy

from ..amino_acids import amino_acids_dict

AMINO_ACIDS = list(amino_acids_dict)


def random_peptides(num, length=9):
    """
    Generate uniformly random peptides (kmers).

    Parameters
    ----------
    num : int
        Number of peptides to return

    length : int
        Length of each peptide

    Returns
    ----------
    list of string

    """
    if num == 0:
        return []
    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            AMINO_ACIDS, size=(int(num), int(length)))
    ]
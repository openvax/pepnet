# Copyright (c) 2017. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from collections import OrderedDict

from .amino_acids import (
    post_translation_modifications_dict,
    amino_acids_dict,
)


class Encoder(object):
    """
    Container for mapping between amino acid letter codes and their full names
    and providing index/hotshot encodings of amino acid sequences.

    The reason we need a class to contain these mappins is that we might want to
    operate on just the common 20 amino acids or a richer set with modifications.
    """
    def __init__(
            self,
            tokens_to_names_dict=amino_acids_dict,
            ignore_ptm_list=[],
            encode_ptm_list=[],
            variable_length_sequences=True,
            add_start_tokens=False,
            add_stop_tokens=False):
        """
        Parameters
        ----------
        tokens_to_names_dict : dict
            Dictionary mapping each amino acid to its name

        ignore_ptm_list : list of str
            List of post-translational modifications we should ignore by mapping
            their tokens onto the original unmodified amino acid.

        encode_ptm_list : list of str
            List of post-translational modifications we should encode.

        variable_length_sequences : bool
            Do we expect to encode peptides of varying lengths? If so, include
            the gap token "-" in the encoder's alphabet.

        add_start_tokens : bool
            Prefix each peptide string with "^"

        add_stop_tokens : bool
            End each peptide string with "$"
        """
        self._tokens_to_names = OrderedDict()
        self.ignore_ptm_list = ignore_ptm_list
        self.encode_ptm_list = encode_ptm_list
        self.variable_length_sequences = variable_length_sequences
        self.add_start_tokens = add_start_tokens
        self.add_stop_tokens = add_stop_tokens

        if self.variable_length_sequences:
            self._add_token("-", "Gap")

        if self.add_start_tokens:
            self._add_token("^", "Start")

        if self.add_stop_tokens:
            self._add_token("$", "Stop")

        for (k, v) in tokens_to_names_dict.items():
            self._add_token(k, v)

        for ptm_token in encode_ptm_list:
            if ptm_token not in post_translation_modifications_dict:
                raise ValueError("Unknown modified amino acid '%s'" % (ptm_token,))
            self._add_token(ptm_token, post_translation_modifications_dict[ptm_token])

        self._uppercase_set = set([])
        for ptm_token in ignore_ptm_list:
            if ptm_token not in post_translation_modifications_dict:
                raise ValueError("Unknown modified amino acid '%s'" % (ptm_token,))
            self._uppercase_set.add(ptm_token)

    def _add_token(self, token, name):
        assert token not in self._tokens_to_names
        assert len(token) == 1
        self._tokens_to_names[token] = name

    def tokens(self):
        """
        Return letters in sorted order, special characters should get indices
        lower than actual amino acids in this order:
            1) "-"
            2) "^"
            3) "$"
        Currently we're enforcing this order by having the _tokens_to_names
        dictionary be an OrderedDict and adding special tokens before amino
        acids in the __init__ method.
        """
        return list(self._tokens_to_names.keys())

    def names(self):
        return [self._tokens_to_names[k] for k in self.tokens()]

    def index_dict(self):
        return {c: i for (i, c) in enumerate(self.tokens())}

    def __getitem__(self, k):
        return self._tokens_to_names[k]

    def __setitem__(self, k, v):
        self._add_token(k, v)

    def __len__(self):
        return len(self.tokens())

    def index_encoding_list(self, peptides):
        index_dict = self.index_dict()
        return [
            [index_dict[amino_acid] for amino_acid in peptide]
            for peptide in peptides
        ]

    def _validate_and_transform_peptides(
            self,
            peptides,
            max_peptide_length=None):
        if max_peptide_length is None:
            max_peptide_length = max(len(p) for p in peptides)

        if self.variable_length_sequences:
            max_observed_length = max(len(p) for p in peptides)
            if max_observed_length > max_peptide_length:
                raise ValueError(
                    "Can't have peptide(s) of length %d and max_peptide_length=%d" % (
                        max_observed_length,
                        max_peptide_length))
        elif any(len(p) != max_peptide_length for p in peptides):
            raise ValueError("Expected all peptides to have length %d" % (
                max_peptide_length))
        # TODO: uppercase PTMs that we're ignoring and make sure no weird
        # characters are in the given sequences

    def encode_indices(
            self,
            peptides,
            max_peptide_length=None):
        """
        Encode a set of equal length peptides as a matrix of their
        amino acid indices.
        """
        peptides = self._validate_and_transform_peptides(peptides, max_peptide_length)
        n_peptides = len(peptides)
        X = np.zeros((n_peptides, max_peptide_length), dtype=int)
        index_dict = self.index_dict()
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                # we're expecting the token '-' to have index 0 so it's
                # OK to only loop until the end of the given sequence
                X[i, j] = index_dict[amino_acid]
        return X

    def encode_onehot(
            self,
            peptides,
            peptide_length):
        """
        Encode a set of equal length peptides as a binary matrix,
        where each letter is transformed into a length 20 vector with a single
        element that is 1 (and the others are 0).
        """
        shape = (len(peptides), peptide_length, len(self))
        index_dict = self.index_dict()
        X = np.zeros(shape, dtype=bool)
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                k = index_dict[amino_acid]
                X[i, j, k] = 1
        return X

    def encode_FOFE(self, sequences, alpha=0.7, bidirectional=False,):
        """
        Implementation of FOFE encoding from:
            A Fixed-Size Encoding Method for Variable-Length Sequences with its
            Application to Neural Network Language Models

        Parameters
        ----------
        sequences : list of strings
        alpha: float, forgetting factor
        bidirectional: boolean, whether to do both a forward pass
                       and a backward pass over the string
        """
        n_seq = len(sequences)
        index_dict = self.index_dict()
        n_symbols = len(index_dict)
        if bidirectional:
            result = np.zeros((n_seq, 2 * n_symbols), dtype=float)
        else:
            result = np.zeros((n_seq, n_symbols), dtype=float)
        for i, seq in enumerate(sequences):
            l = len(seq)
            for j, sj in enumerate(seq):
                result[i, index_dict[sj]] += alpha ** (l - j - 1)
                if bidirectional:
                    result[i, n_symbols + index_dict[sj]] += alpha ** j
        return result

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

from .amino_acids import amino_acids_dict


class Encoder(object):
    """
    Container for mapping between amino acid letter codes and their full names
    and providing index/hotshot encodings of amino acid sequences.

    The reason we need a class to contain these mappins is that we might want to
    operate on just the common 20 amino acids or a richer set with modifications.
    """
    def __init__(
            self,
            amino_acid_alphabet=amino_acids_dict,
            variable_length_sequences=True,
            add_start_tokens=False,
            add_stop_tokens=False):
        """
        Parameters
        ----------
        tokens_to_names_dict : dict
            Dictionary mapping each amino acid to its name

        variable_length_sequences : bool
            Do we expect to encode peptides of varying lengths? If so, include
            the gap token "-" in the encoder's alphabet.

        add_start_tokens : bool
            Prefix each peptide string with "^"

        add_stop_tokens : bool
            End each peptide string with "$"
        """
        self._tokens_to_names = OrderedDict()
        self._index_dict = {}

        self.amino_acid_alphabet = amino_acid_alphabet
        self.variable_length_sequences = variable_length_sequences
        self.add_start_tokens = add_start_tokens
        self.add_stop_tokens = add_stop_tokens

        if self.variable_length_sequences:
            self._add_token("-", "Gap")

        if self.add_start_tokens:
            self._add_token("^", "Start")

        if self.add_stop_tokens:
            self._add_token("$", "Stop")

        for (k, v) in amino_acid_alphabet.items():
            self._add_token(k, v)

    def _add_token(self, token, name):
        assert len(token) == 1, "Invalid token '%s'" % (token,)
        assert token not in self._index_dict
        assert token not in self._tokens_to_names
        self._index_dict[token] = len(self._index_dict)
        self._tokens_to_names[token] = name

    def prepare_sequences(self, peptides, padded_peptide_length=None):
        """
        Add start/stop tokens to each peptide (if required) and
        if padded_peptide_length is provided then pad each peptide to
        be the same length using the gap token '-'.
        """
        if self.add_start_tokens:
            peptides = ["^" + p for p in peptides]
            if padded_peptide_length:
                padded_peptide_length += 1

        if self.add_stop_tokens:
            peptides = [p + "$" for p in peptides]
            if padded_peptide_length:
                padded_peptide_length += 1

        if padded_peptide_length:
            peptides = [
                p + "-" * (padded_peptide_length - len(p))
                for p in peptides
            ]
        return peptides

    @property
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

    @property
    def amino_acid_names(self):
        return [self._tokens_to_names[k] for k in self.tokens]

    @property
    def index_dict(self):
        return self._index_dict

    def __getitem__(self, k):
        return self._tokens_to_names[k]

    def __setitem__(self, k, v):
        self._add_token(k, v)

    def __len__(self):
        return len(self.tokens)

    def _validate_peptide_lengths(
            self,
            peptides,
            max_peptide_length=None):
        if max_peptide_length is None:
            max_peptide_length = max(len(p) for p in peptides)

        if self.variable_length_sequences:
            max_observed_length = max(len(p) for p in peptides)
            if max_observed_length > max_peptide_length:
                example = [p for p in peptides if len(p) == max_observed_length][0]
                raise ValueError(
                    "Can't have peptide(s) of length %d and max_peptide_length = %d (example '%s')" % (
                        max_observed_length,
                        max_peptide_length,
                        example))
        elif any(len(p) != max_peptide_length for p in peptides):
            example = [p for p in peptides if len(p) != max_peptide_length][0]
            raise ValueError("Expected all peptides to have length %d, '%s' has length %d" % (
                max_peptide_length,
                example,
                len(example)))
        return max_peptide_length

    def _validate_and_prepare_peptides(self, peptides, max_peptide_length=None):
        max_peptide_length = self._validate_peptide_lengths(
            peptides, max_peptide_length)
        peptides = self.prepare_sequences(peptides)
        # did we add start tokens to each sequence?
        max_peptide_length += self.add_start_tokens
        # did we add stop tokens to each sequence?
        max_peptide_length += self.add_stop_tokens
        return peptides, max_peptide_length

    def encode_index_lists(self, peptides):
        # don't try to do length validation since we're allowed to have
        # multiple peptide lengths
        peptides = self.prepare_sequences(peptides)
        index_dict = self.index_dict
        return [
            [index_dict[amino_acid] for amino_acid in peptide]
            for peptide in peptides
        ]

    def encode_index_array(
            self,
            peptides,
            max_peptide_length=None):
        """
        Encode a set of equal length peptides as a matrix of their
        amino acid indices.
        """
        peptides, max_peptide_length = self._validate_and_prepare_peptides(
            peptides, max_peptide_length)
        n_peptides = len(peptides)
        X_index = np.zeros((n_peptides, max_peptide_length), dtype=int)
        index_dict = self.index_dict
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                # we're expecting the token '-' to have index 0 so it's
                # OK to only loop until the end of the given sequence
                X_index[i, j] = index_dict[amino_acid]
        return X_index

    def encode_onehot(
            self,
            peptides,
            max_peptide_length=None):
        """
        Encode a set of equal length peptides as a binary matrix,
        where each letter is transformed into a length 20 vector with a single
        element that is 1 (and the others are 0).
        """
        peptides, max_peptide_length = self._validate_and_prepare_peptides(
            peptides, max_peptide_length)
        index_dict = self.index_dict
        n_symbols = len(index_dict)
        X = np.zeros((len(peptides), max_peptide_length, n_symbols), dtype=bool)
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                X[i, j, index_dict[amino_acid]] = 1
        return X

    def encode_FOFE(self, peptides, alpha=0.7, bidirectional=False):
        """
        Implementation of FOFE encoding from:
            A Fixed-Size Encoding Method for Variable-Length Sequences with its
            Application to Neural Network Language Models

        Parameters
        ----------
        peptides : list of strings

        alpha: float
            Forgetting factor

        bidirectional: boolean
            Whether to do both a forward pass and a backward pass over each
            peptide
        """
        # don't try to do length validation since we're allowed to have
        # multiple peptide lengths in a FOFE encoding
        peptides = self.prepare_sequences(peptides)
        n_peptides = len(peptides)
        index_dict = self.index_dict
        n_symbols = len(index_dict)
        if bidirectional:
            result = np.zeros((n_peptides, 2 * n_symbols), dtype=float)
        else:
            result = np.zeros((n_peptides, n_symbols), dtype=float)
        for i, p in enumerate(peptides):
            l = len(p)
            for j, amino_acid in enumerate(p):
                aa_idx = index_dict[amino_acid]
                result[i, aa_idx] += alpha ** (l - j - 1)
                if bidirectional:
                    result[i, n_symbols + aa_idx] += alpha ** j
        return result

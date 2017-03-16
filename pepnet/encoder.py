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

class Encoder(object):
    """
    Container for mapping between amino acid letter codes and their full names
    and providing index/hotshot encodings of amino acid sequences.

    The reason we need a class to contain these mappins is that we might want to
    operate on just the common 20 amino acids or a richer set with modifications.
    """

    def __init__(
            self,
            letters_to_names,
            variable_length=True,
            add_start_tokens=False,
            add_stop_tokens=False):

        self.letters_to_names = {}
        for (k, v) in letters_to_names.items():
            self.add(k, v)

        self.variable_length = variable_length
        self.add_start_tokens = add_start_tokens
        self.add_stop_tokens = add_stop_tokens

        if self.variable_length:
            assert "-" not in self.letters_to_names
            self.add("-", "Gap")

        if self.add_start_tokens:
            assert "^" not in self.letters_to_names
            self.add("^", "Start")

        if self.add_stop_tokens:
            assert "$" not in self.letters_to_names
            self.add("$", "Stop")

    def add(self, letter, name):
        assert letter not in self.letters_to_names
        assert len(letter) == 1
        self.letters_to_names[letter] = name

    def letters(self):
        """
        Return letters in sorted order, special characters should get indices
        lower than actual amino acids in this order:
            1) "-"
            2) "^"
            3) "$"
        """
        return list(sorted(
            self.letters_to_names.keys(),
            key=lambda x: (x != "-", x != "^", x != "$", x)))

    def names(self):
        return [self.letters_to_names[k] for k in self.letters()]

    def index_dict(self):
        return {c: i for (i, c) in enumerate(self.letters())}

    def copy(self):
        return Encoder(self.letters_to_names)

    def __getitem__(self, k):
        return self.letters_to_names[k]

    def __setitem__(self, k, v):
        self.add(k, v)

    def __len__(self):
        return len(self.letters_to_names)

    def index_encoding_list(self, peptides):
        index_dict = self.index_dict()
        return [
            [index_dict[amino_acid] for amino_acid in peptide]
            for peptide in peptides
        ]

    def encode_indices(
            self,
            peptides,
            max_peptide_length=None):
        """
        Encode a set of equal length peptides as a matrix of their
        amino acid indices.
        """
        if self.variable_length:
            if max_peptide_length is None:
                max_peptide_length = max(len(p) for p in peptides)
        else:
            offset = 0
            max_peptide_length = 0
            if any(len(p))
        X = np.zeros((len(peptides), peptide_length), dtype=int)
        index_dict = self.index_dict()
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
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

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

from __future__ import (
    print_function,
    division,
    absolute_import,
)

from .encoder import Encoder

###
# 20 common amino acids
###

amino_acids = {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}
amino_acid_encoder = Encoder(amino_acids)
amino_acid_letters = amino_acid_encoder.letters()

###
# Allow 'X' for amino acids whose identity is not known
###
amino_acids_with_unknown = amino_acids.copy()
amino_acids_with_unknown["X"] = "Unknown"
amino_acid_encoder_with_unknown = Encoder(amino_acids_with_unknown)
amino_acid_letters_with_unknown = amino_acid_encoder_with_unknown.letters()

###
# In cases where a sequence might have both unknown AAs and a gap
# we need to distinguish the two characters.
###
amino_acids_with_unknown_and_gap = amino_acids_with_unknown.copy()
amino_acids_with_unknown_and_gap["-"] = "Gap"
amino_acid_encoder_with_unknown_and_gap = Encoder(amino_acids_with_unknown_and_gap)
amino_acid_letters_with_unknown_and_gap = \
    amino_acid_encoder_with_unknown_and_gap.letters()


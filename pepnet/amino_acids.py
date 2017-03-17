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

###
# 20 common amino acids
###
amino_acids_dict = {
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

###
# Including 7 post-translational modifications commonly found by mass spec
###
post_translation_modifications_dict = {
    "s": "Phospho-Serine",
    "t": "Phospho-Threonine",
    "y": "Phospho-Tyrosine",
    "c": "Cystine",
    "m": "Methionine sulfoxide",
    "q": "Pyroglutamate",
    "n": "Pyroglutamic acid"
}

def amino_acids_with_ptms(ptms=[]):
    """
    Generate a dictionary mapping amino acids to their names, along
    with a specific set of post-translational modifications.
    """
    amino_acids_with_ptms_dict = amino_acids_dict.copy()
    for ptm in ptms:
        name = post_translation_modifications_dict[ptm]
        amino_acids_with_ptms_dict[ptm] = name
    return amino_acids_with_ptms_dict

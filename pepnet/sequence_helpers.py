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

from random import shuffle
import numpy as np
from collections import Counter

def _group_sequences_by_kmers(seqs, k=9):
    kmer_to_group_dict = {}
    group_list = []
    for seq in seqs:
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        group = None
        for kmer in kmers:
            if kmer in kmer_to_group_dict:
                group = kmer_to_group_dict[kmer]
                break
        if not group:
            group = set([seq])
            group_list.append(group)
        else:
            group.add(seq)
        for kmer in kmers:
            kmer_to_group_dict[kmer] = group
    return group_list

def random_iter(seq):
    shuffled = list(seq)
    shuffle(shuffled)
    return iter(shuffled)

def _group_similar_sequences_and_flatten(seqs, k=9):
    groups = _group_sequences_by_kmers(seqs, k=k)
    seq_list = []
    group_id_list = []
    for i, group in enumerate(random_iter(groups)):
        for seq in random_iter(group):
            seq_list.append(seq)
            group_id_list.append(i)
    return seq_list, np.array(group_id_list)

def group_similar_sequences(seqs, k=9):
    """
    Group sequences by kmer content. Returns list of sequences, group IDs for
    each sequence, and weights that are inversely proportional to size of each
    group.
    """
    seqs, group_ids = _group_similar_sequences_and_flatten(seqs=seqs, k=k)
    weights = np.ones(len(seqs))
    counts = Counter()
    for group_id in group_ids:
        counts[group_id] += 1
    for i, group_id in enumerate(group_ids):
        weights[i] = 1.0 / counts[group_id]
    return seqs, group_ids, weights

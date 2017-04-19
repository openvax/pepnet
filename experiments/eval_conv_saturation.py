import os
import csv
from random import shuffle
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from pepnet import Predictor, SequenceInput, Output

def make_predictors(
        widths=[9],
        layer_sizes=[16],
        n_conv_layers=[2],
        conv_dropouts=[0]):
    return {
        (width, layer_size, n_layers, dropout): Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[1, 3, width],
                n_conv_layers=n_layers,
                conv_output_dim=layer_size,
                conv_dropout=dropout,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid"))
        for width in widths
        for layer_size in layer_sizes
        for n_layers in n_conv_layers
        for dropout in conv_dropouts
    }


def make_decoy_set(hits, multiple=10):
    from collections import Counter
    import pyensembl
    proteins_dict = pyensembl.ensembl_grch38.protein_sequences.fasta_dictionary
    protein_list = list(proteins_dict.values())
    lengths = Counter()
    for hit in hits:
        lengths[len(hit)] += 1

    decoys = set([])
    n_proteins = len(protein_list)
    for length, count in lengths.items():
        for protein_idx in np.random.randint(low=0, high=n_proteins, size=count * multiple):
            protein = protein_list[protein_idx]
            if len(protein) < length:
                continue

            i = np.random.randint(low=0, high=len(protein) - length + 1, size=1)[0]
            peptide = protein[i:i + length]
            if "X" in peptide or "U" in peptide or "*" in peptide:
                continue
            decoys.add(peptide)
    return decoys

def group_similar_sequences(seqs, k=9):
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

def assign_group_ids_to_sequences(seqs, k=9):
    groups = group_similar_sequences(seqs, k=k)
    seq_list = []
    group_id_list = []
    for i, group in enumerate(random_iter(groups)):
        for seq in random_iter(group):
            seq_list.append(seq)
            group_id_list.append(i)
    return seq_list, np.array(group_id_list)

def assign_group_ids_and_weights(seqs, k=9):
    seqs, group_ids = assign_group_ids_to_sequences(seqs=seqs, k=k)
    weights = np.ones(len(seqs))
    counts = Counter()
    for group_id in group_ids:
        counts[group_id] += 1
    for i, group_id in enumerate(group_ids):
        weights[i] = 1.0 / counts[group_id]
    return seqs, group_ids, weights

def load_hits(filename=None):
    if not filename:
        filename = os.environ["CLASS_II_DATA"]
    df = pd.read_excel(filename)
    hits = {}
    for allele in df.columns:
        hits[allele] = [
            s.upper() for s in df[allele]
            if isinstance(s, str) and len(s) > 0 and "X" not in s]
        print("Loaded %d hits for %s" % (len(hits[allele]), allele))
    return hits


if __name__ == "__main__":
    hits = load_hits()

    n_splits = 3

    cv = GroupKFold(n_splits=n_splits)

    with open('scores_conv_saturation_larger_models.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_training",
            "width", "layer_size", "n_layers", "dropout",
            "allele", "fold", "auc"])
        writer.writeheader()
        for allele, allele_hits in hits.items():
            print(allele)

            allele_hits, kmer_group_ids, sample_weights = \
                assign_group_ids_and_weights(allele_hits, k=9)

            print("Split %s hits into %d groups" % (
                allele,
                max(kmer_group_ids) + 1,))
            for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(X=allele_hits, y=None, groups=kmer_group_ids)):
                all_train_hits = [allele_hits[i] for i in train_idx]
                test_hits = [allele_hits[i] for i in test_idx]
                test_decoys = make_decoy_set(test_hits)

                test = list(test_hits) + list(test_decoys)
                y_test = [True] * len(test_hits) + [False] * len(test_decoys)
                test_weights = np.ones(len(y_test))
                test_weights[:len(test_hits)] = sample_weights[test_idx]
                for n_training in np.linspace(50, len(all_train_hits), num=10, dtype=int):
                    epochs = min(10**4 // n_training, 50)
                    train_hits = all_train_hits[:n_training]
                    train_decoys = make_decoy_set(train_hits)
                    train = list(train_hits) + list(train_decoys)
                    y_train = [True] * len(train_hits) + [False] * len(train_decoys)
                    train_weights = np.ones(len(y_train))
                    train_weights[:len(train_hits)] = sample_weights[train_idx[:n_training]]
                    predictor_dict = make_predictors()
                    for key in sorted(predictor_dict.keys()):
                        model = predictor_dict[key]
                        (width, layer_size, n_conv_layers, dropout) = key
                        row_dict = {
                            "width": width,
                            "layer_size": layer_size,
                            "n_layers": n_conv_layers,
                            "dropout": dropout,
                        }
                        print("==> Training %s" % (row_dict,))
                        model.fit(train, y_train, sample_weight=train_weights, epochs=epochs)
                        pred = model.predict(test)
                        auc = roc_auc_score(y_true=y_test, y_score=pred, sample_weight=test_weights)
                        print("==> %s %d/%d %s: %0.4f" % (
                            allele, fold_idx + 1, n_splits, row_dict, auc))
                        row_dict["allele"] = allele
                        row_dict["fold"] = fold_idx
                        row_dict["auc"] = auc
                        row_dict["n_training"] = n_training
                        writer.writerow(row_dict)
                        f.flush()

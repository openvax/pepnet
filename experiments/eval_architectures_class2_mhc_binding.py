import os
import collections

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from pepnet import Predictor, SequenceInput, Output

def make_predictors():
    return {
        "pool": Predictor(
            inputs=SequenceInput(name="peptide", length=22, variable_length=True, global_pooling=True),
            outputs=Output(1, activation="sigmoid")),
        "rnn": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                rnn_layer_sizes=[32]),
            outputs=Output(1, activation="sigmoid")),
        "rnn2": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                rnn_layer_sizes=[32, 32]),
            outputs=Output(1, activation="sigmoid")),
        "conv-pool": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[9],
                conv_output_dim=16,
                conv_dropout=0.1,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid")),
        "conv2-pool": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[9],
                n_conv_layers=2,
                conv_output_dim=16,
                conv_dropout=0.1,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid")),
        "conv-rnn": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[9],
                conv_output_dim=16,
                conv_dropout=0.1,
                rnn_layer_sizes=[32]),
            outputs=Output(1, activation="sigmoid")),
        "multiconv-pool": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[3, 9],
                conv_output_dim=16,
                conv_dropout=0.1,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid")),
        "multiconv2-pool": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[3, 9],
                n_conv_layers=2,
                conv_output_dim=16,
                conv_dropout=0.1,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid")),
        "multiconv-rnn": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[3, 9],
                conv_output_dim=16,
                conv_dropout=0.1,
                rnn_layer_sizes=[32]),
            outputs=Output(1, activation="sigmoid")),
        "multiconv2-rnn": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[3, 9],
                n_conv_layers=2,
                conv_output_dim=16,
                conv_dropout=0.1,
                rnn_layer_sizes=[32]),
            outputs=Output(1, activation="sigmoid")),
        "multiconv2-rnn2": Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[3, 9],
                n_conv_layers=2,
                conv_output_dim=16,
                conv_dropout=0.1,
                rnn_layer_sizes=[32, 32]),
            outputs=Output(1, activation="sigmoid"))
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


if __name__ == "__main__":
    df = pd.read_excel(os.environ["CLASS_II_DATA"])
    hits = {}
    for col in df.columns:
        hits[col] = [s.upper() for s in df[col] if isinstance(s, str) and len(s) > 0 and "X" not in s]
        print(col, len(hits[col]))
    n_splits = 3
    epochs = 30
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    scores = {
        predictor_name: collections.defaultdict(list)
        for predictor_name in make_predictors().keys()
    }
    for allele in df.columns:
        print(allele)
        curr_hits = hits[allele]
        # stratify by short/medium/long
        peptide_length_groups = [0 if len(s) < 14 else (2 if len(s) > 17 else 1) for s in curr_hits]
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X=curr_hits, y=peptide_length_groups)):
            train_hits = [curr_hits[i] for i in train_idx]
            train_decoys = make_decoy_set(train_hits)
            train = list(train_hits) + list(train_decoys)
            y_train = [True] * len(train_hits) + [False] * len(train_decoys)

            test_hits = [curr_hits[i] for i in test_idx]
            test_decoys = make_decoy_set(test_hits)
            test = list(test_hits) + list(test_decoys)
            y_test = [True] * len(test_hits) + [False] * len(test_decoys)
            for model_name, model in make_predictors().items():
                print("==> Training %s" % model_name)
                model.fit(train, y_train, epochs=epochs)
                pred = model.predict(test)
                auc = roc_auc_score(y_true=y_test, y_score=pred)
                print("==> %s %d/%d %s: %0.4f" % (
                    allele, fold_idx + 1, n_splits, model_name, auc))
                scores[model_name][allele].append(auc)
            with open('scores.csv', 'w') as f:
                f.write("model,allele,fold,auc\n")
                for m in scores.keys():
                        for auc in scores[m][allele]:
                            f.write("%s,%s,%d,%0.4f\n" % (m, allele, fold_idx, auc))

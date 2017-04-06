import os


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from pepnet import Predictor, SequenceInput, Output

def make_predictors(
        widths=[8, 9, 10],
        layer_sizes=[4, 16, 32],
        n_conv_layers=[1, 2],
        conv_dropouts=[0, 0.25]):
    return {
        "width=%d, layer_size=%d, n_layers=%d, conv=%0.2f" % (
            width, layer_size, n_layers, dropout): Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=22,
                variable_length=True,
                conv_filter_sizes=[width],
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


if __name__ == "__main__":
    df = pd.read_excel(os.environ["CLASS_II_DATA"])
    hits = {}
    for col in df.columns:
        hits[col] = [s.upper() for s in df[col] if isinstance(s, str) and len(s) > 0 and "X" not in s]
        print(col, len(hits[col]))
    n_splits = 3
    epochs = 30
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    with open('scores_conv.csv', 'w') as f:
        f.write("model,allele,fold,auc\n")
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
                predictor_dict = make_predictors()
                for model_name in sorted(predictor_dict.keys()):
                    model = predictor_dict[model_name]
                    print("==> Training %s" % model_name)
                    model.fit(train, y_train, epochs=epochs)
                    pred = model.predict(test)
                    auc = roc_auc_score(y_true=y_test, y_score=pred)
                    print("==> %s %d/%d %s: %0.4f" % (
                        allele, fold_idx + 1, n_splits, model_name, auc))
                    f.write("%s,%s,%d,%0.4f\n" % (model_name, allele, fold_idx, auc))
                    f.flush()

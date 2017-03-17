# pepnet
Neural networks for amino acid sequences

## Fixed-length peptide input represented by one-shot binary vectors

```python
from pepnet.feed_forward import make_fixed_length_hotshot_network

# make a model whose input is a single amino acid
model = make_fixed_length_hotshot_network(peptide_length=1, n_symbols=20)
X = np.zeros((2, 20), dtype=bool)
X[0, 0] = True
X[1, 5] = True
Y = np.array([True, False])
model.fit(X, Y)
```


## Fixed-length peptide input represented by learned amino acid embeddings
```python
from pepnet.feed_forward import make_fixed_length_embedding_network
model = make_fixed_length_embedding_network(
    peptide_length=1, n_symbols=20, embedding_output_dim=40)
X = np.array([[9], [7]])
Y = np.array([True, False])
model.fit(X, Y)
```


## Networks with variable-length peptides and fixed-length context

```python
from pepnet.sequence_context import make_variable_length_model_with_fixed_length_context
from pepnet.encoder import Encoder

model = make_variable_length_model_with_fixed_length_context(
    n_upstream=1,
    n_downstream=1,
    max_peptide_length=3)
encoder = Encoder()
X_peptide = encoder.encode_index_array([
    "SYF",
    "QQ",
    "C",
    "GLL"], max_peptide_length=3)

input_dict = {
    "upstream": encoder.encode_index_array(["Q", "A", "L", "I"]),
    "downstream": encoder.encode_index_array(["S"] * 4),
    "peptide": X_peptide
}
Y = np.array([True, False, True, False])
model.fit(input_dict, Y)
```
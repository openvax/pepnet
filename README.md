# pepnet
Neural networks for amino acid sequences

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
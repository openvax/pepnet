from pepnet.sequence_context import make_variable_length_model_with_fixed_length_context
from pepnet.encoder import Encoder
import numpy as np

def test_model_with_fixed_length_context():
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
    X_upstream = encoder.encode_index_array(["Q", "A", "L", "I"])
    X_downstream = encoder.encode_index_array(["S"] * 4)
    Y = np.array([True, False, True, False])
    input_dict = {
        "upstream": X_upstream,
        "downstream": X_downstream,
        "peptide": X_peptide
    }
    model.fit(input_dict, Y, epochs=20)
    Y_pred = model.predict(input_dict)
    assert (Y == (Y_pred[:, 0] > 0.5)).all(), (Y, Y_pred)

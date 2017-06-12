from pepnet import Predictor, SequenceInput, Output

import numpy as np

def test_model_with_fixed_length_context():
    model = Predictor(
        inputs={
            "upstream": SequenceInput(length=1, variable_length=False),
            "downstream": SequenceInput(length=1, variable_length=False),
            "peptide": SequenceInput(length=3, variable_length=True)},
        outputs=Output(1, activation="sigmoid"))

    Y = np.array([True, False, True, False])
    input_dict = {
        "upstream": ["Q", "A", "L", "I"],
        "downstream": ["S"] * 4,
        "peptide": ["SYF", "QQ", "C", "GLL"]
    }
    model.fit(input_dict, Y, epochs=20)
    Y_pred = model.predict(input_dict)
    assert (Y == (Y_pred > 0.5)).all(), (Y, Y_pred)

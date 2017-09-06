from pepnet import Predictor, DiscreteInput, Output

def test_discrete_input_with_str_tokens():
    pred = Predictor(
        inputs=DiscreteInput(choices=["x", "y", "z"], embedding_dim=2),
        outputs=Output(1, "sigmoid"))
    pred.fit(["x", "x", "y", "z"], [0, 0, 0.5, 1.0], epochs=20)
    assert pred.predict(["x"]) < pred.predict(["z"])
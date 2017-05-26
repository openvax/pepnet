from pepnet.nn_helpers import highway_layers
from keras.layers import Input, Embedding, Dense, Flatten
from keras.models import Model
import numpy as np

def test_highway_layers():
    n_highway_layers = 5
    x = Input(shape=(8,), dtype="int32")
    v = Embedding(input_dim=2, output_dim=10)(x)
    v = Flatten()(v)
    assert hasattr(v, "_keras_shape")
    v = highway_layers(v, n_layers=n_highway_layers)
    output = Dense(1)(v)
    model = Model(inputs=[x], outputs=[output])
    assert len(model.layers) > n_highway_layers * 3
    x = np.array([
        [1] + [0] * 7,
        [0] * 8,
        [0] * 7 + [1]])
    y = np.array([0, 1, 0])
    model.compile("rmsprop", "mse")
    model.fit(x, y, epochs=10)
    pred = model.predict(x)
    mean_diff = np.abs(pred - y).mean()
    assert mean_diff < 0.5, pred

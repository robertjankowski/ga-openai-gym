import os

from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass
import pandas as pd


@dataclass
class Shape:
    input: int
    output: int


class MultiTaskModel:
    def __init__(self, first_shape: Shape, second_shape: Shape, *hidden_layers):
        self.model = self._build_model(first_shape, second_shape, *hidden_layers)
        self._compile_model()

    @staticmethod
    def _build_model(first_shape: Shape, second_shape: Shape, *hidden_layers):
        first_input = keras.Input(shape=first_shape.input, name="first_input")
        second_input = keras.Input(shape=second_shape.input, name="second_input")

        x = layers.concatenate([first_input, second_input])
        for hl in hidden_layers:
            x = layers.Dense(hl)(x)

        first_output = layers.Dense(first_shape.output, name="first_output")(x)
        second_output = layers.Dense(second_shape.output, name="second_output")(x)

        return keras.Model(inputs=[first_input, second_input], outputs=[first_output, second_output])

    def _compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss={
                "first_output": keras.losses.MeanSquaredError(),
                "second_output": keras.losses.MeanSquaredError()
            },
            loss_weights=[1.0, 1.0]
        )

    def fit(self, first_data: dict, second_data: dict, epochs: int, batch_size: int, callbacks: list = None):
        self.model.fit(first_data, second_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def plot_model(self, output_path, show_shapes=False):
        keras.utils.plot_model(self.model, output_path, show_shapes=show_shapes)

    def load_weights(self, checkpoint_path: str):
        self.model.load_weights(checkpoint_path)


def load_dataset(features_path: str, labels_path: str, use_columns=None):
    X = pd.read_csv(features_path, header=None, usecols=use_columns)
    y = pd.read_csv(labels_path, header=None)
    return X.values, y.values


def checkpoint_callback(path: str):
    checkpoint_dir = os.path.dirname(path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=path,
                                                  save_weights_only=True,
                                                  verbose=1)
    return cp_callback


if __name__ == '__main__':
    cartpole_shape = Shape(4, 1)
    bipedalwalker_shape = Shape(24, 4)

    hidden_sizes = [12, 20, 4]
    model = MultiTaskModel(cartpole_shape, bipedalwalker_shape, *hidden_sizes)
    # model.plot_model("../../docs/multitask_learning_example_architecture.png", show_shapes=True)

    X_bipedalwalker, y_bipedalwalker = load_dataset(
        "../../models/bipedalwalker/generated_data/features_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_subset.csv",
        "../../models/bipedalwalker/generated_data/labels_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_subset.csv"
    )
    X_cartpole, y_cartpole = load_dataset(
        "../../models/cartpole/features_data_cartpole12-27-2019_20-29_NN=MLPIndividual_POPSIZE=100_GEN=20_PMUTATION_0_NRUNS=500.csv",
        "../../models/cartpole/labels_data_cartpole12-27-2019_20-29_NN=MLPIndividual_POPSIZE=100_GEN=20_PMUTATION_0_NRUNS=500.csv"
    )
    X_cartpole = X_cartpole[:X_bipedalwalker.shape[0]]
    y_cartpole = y_cartpole[:X_bipedalwalker.shape[0]]

    model.fit(
        {"first_input": X_cartpole, "second_input": X_bipedalwalker},
        {"first_output": y_cartpole, "second_output": y_bipedalwalker},
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint_callback("multitask-model-test/model")]
    )

import numpy as np

from nn.activation_function import ActivationFunctions
from nn.base_nn import NeuralNetwork

ReLU = ActivationFunctions.ReLU
Sigmoid = ActivationFunctions.Sigmoid


class MLP(NeuralNetwork):
    """
    Implementation of the multilayer perception (1 hidden layer) in Numpy
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases_1 = np.random.randn(self.hidden_size)
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size)
        self.biases_2 = np.random.rand(self.output_size)

    def forward(self, x: np.array) -> np.array:
        x = x @ self.weights_1 + self.biases_1
        x = ReLU(x)
        x = x @ self.weights_2 + self.biases_2
        return Sigmoid(x)

    def get_weights_biases(self) -> np.array:
        w_1 = self.weights_1.flatten()
        w_2 = self.weights_2.flatten()
        return np.concatenate((w_1, self.biases_1, w_2, self.biases_2), axis=0)

    def update_weights_biases(self, weights_biases: np.array) -> None:
        w_1, b_1, w_2, b_2 = np.split(weights_biases,
                                      [self.weights_1.size, self.weights_1.size + self.biases_1.size,
                                       self.weights_1.size + self.biases_1.size + self.weights_2.size])
        self.weights_1 = np.resize(w_1, (self.input_size, self.hidden_size))
        self.biases_1 = b_1
        self.weights_2 = np.resize(w_2, (self.hidden_size, self.output_size))
        self.biases_2 = b_2

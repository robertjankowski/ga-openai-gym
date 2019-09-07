import numpy as np


class NeuralNetwork:
    def __init__(self, d_in, h, d_out):
        """
        :param d_in: Input size
        :param h: No. of hidden nodes
        :param d_out: Output shape
        """
        self.d_in = d_in
        self.h = h
        self.d_out = d_out
        self.weights_1 = np.random.randn(self.d_in, self.h)
        self.biases_1 = np.random.randn(self.h)
        self.weights_2 = np.random.randn(self.h, self.d_out)
        self.biases_2 = np.random.rand(self.d_out)

    def forward(self, x: np.array) -> np.array:
        x = x @ self.weights_1 + self.biases_1
        x = self.ReLu(x)
        x = x @ self.weights_2 + self.biases_2
        return self.sigmoid(x)

    @staticmethod
    def ReLu(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_weights_biases(self) -> np.array:
        """
        :return: list of weights and biases
        """
        w_1 = self.weights_1.flatten()
        w_2 = self.weights_2.flatten()
        return np.concatenate((w_1, self.biases_1, w_2, self.biases_2), axis=0)

    def update_weights_biases(self, new_weights_biases: np.array):
        """
        :param new_weights_biases: List of weights and biases
        :return: None -> update old weights and biases
        """
        w_1, b_1, w_2, b_2 = np.split(new_weights_biases,
                                      [self.weights_1.size, self.weights_1.size + self.biases_1.size,
                                       self.weights_1.size + self.biases_1.size + self.weights_2.size])
        self.weights_1 = np.resize(w_1, (self.d_in, self.h))
        self.biases_1 = b_1
        self.weights_2 = np.resize(w_2, (self.h, self.d_out))
        self.biases_2 = b_2

    def load(self, file):
        """
        :param file: File with array of weights
        :return:
        """
        self.update_weights_biases(np.load(file))

import numpy as np


class ActivationFunctions:
    @staticmethod
    def ReLU(x):
        return np.maximum(x, 0)

    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def LeakyReLU(x):
        return x if x > 0 else 0.01 * x

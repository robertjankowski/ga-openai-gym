from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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


class MLPTorch(nn.Module, NeuralNetwork):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, p=0.1):
        super(MLPTorch, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout = nn.Dropout(p=p)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x) -> torch.Tensor:
        output = torch.relu(self.linear1(x))
        output = torch.relu(self.linear2(output))
        output = self.dropout(output)
        output = torch.tanh(self.linear3(output))
        return output

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)


class DeepMLPTorch(nn.Module, NeuralNetwork):
    def __init__(self, input_size, output_size, *hidden_sizes):
        super(DeepMLPTorch, self).__init__()
        assert len(hidden_sizes) >= 1
        self.input_size = input_size
        self.output_size = output_size
        self.linear_layers = nn.ModuleList()
        for in_size, out_size in zip([input_size] + [*hidden_sizes], [*hidden_sizes] + [output_size]):
            self.linear_layers.append(nn.Linear(in_size, out_size))

    def forward(self, x) -> torch.Tensor:
        output = x
        for layers in self.linear_layers[:-1]:
            output = layers(output)
        return torch.tanh(self.linear_layers[-1](output))

    def get_weights_biases(self) -> np.array:
        # return self.state_dict()
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        # self.load_state_dict(weights_biases)
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)

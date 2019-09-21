from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch

from nn.base_nn import NeuralNetwork


class RNN(nn.Module, NeuralNetwork):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RNN, self).__init__()
        self.hidden_size1 = hidden_size1

        self.i2h = nn.Linear(input_size + hidden_size1, hidden_size2)
        self.hidden_combined_layer = nn.Linear(hidden_size2, hidden_size1)
        self.output_combined_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, input, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((input, hidden), 0)
        combined = torch.relu(self.i2h(combined))
        hidden = self.hidden_combined_layer(combined)
        output = nn.Tanh()(self.output_combined_layer(combined))
        return output, hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(self.hidden_size1)

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

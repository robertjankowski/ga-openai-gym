from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from nn.base_nn import NeuralNetwork


class ConvNet(nn.Module, NeuralNetwork):
    def __init__(self):
        """
        Input shape: (3, 96, 96)
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=3, padding=0)

        self.fc1 = nn.Linear(12 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # (3, 96, 96) -> (12, 96, 96)
        x = torch.relu(self.conv1(x))

        # (12, 96, 96) -> (12, 32, 32)
        x = self.pool(x)

        # (12, 32, 32) -> (1, 12288)
        x = x.view(-1, 12 * 32 * 32)

        # (1, 12288) -> (1, 64)
        x = torch.relu(self.fc1(x))

        # (1, 64) -> (1, 3)
        x = self.fc2(x)[0]

        # [s, t, b]
        # s = [-1, 1]; t = [0, 1]; b = [0, 1]
        steering_angle = torch.tanh(x[0])
        throttle = torch.sigmoid(x[1])
        use_break = torch.sigmoid(x[2])
        return torch.tensor([steering_angle, throttle, use_break])

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

import gym
import torch

from nn.mlp import MLP, MLPTorch
from nn.rnn import RNN


def test_mlp():
    global observation
    for _ in range(300):
        env.render()
        action = mlp.forward(observation)
        observation, reward, done, _ = env.step(action)
        if done:
            break


def test_mlp_torch():
    global observation
    for _ in range(300):
        env.render()
        observation = torch.from_numpy(observation).float()
        action = mlp_torch.forward(observation)
        action = action.detach().numpy()
        observation, reward, done, _ = env.step(action)
        if done:
            break


def test_rnn():
    global observation
    hidden = rnn.init_hidden()
    for _ in range(300):
        env.render()
        observation = torch.from_numpy(observation).float()
        action, hidden = rnn.forward(observation, hidden)
        action = action.detach().numpy()
        # action = np.nan_to_num(action)
        observation, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)
    observation = env.reset()

    INPUT_SIZE = 24
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4

    rnn = RNN(INPUT_SIZE, 32, 16, OUTPUT_SIZE)
    rnn.load('../../../models/bipedalwalker/09-17-2019_18-05_NN=RNNIndividual_POPSIZE=10_GEN=10_PMUTATION_0'
             '.2_PCROSSOVER_0.9.npy')
    test_rnn()

    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # mlp.load("../../../models/bipedalwalker/09-07-2019_16-34-56_POPSIZE=30_GEN=5_MUTATION_0.609-07-2019_16-34"
    #          "-56_POPSIZE=30_GEN=5_MUTATION_0.6.npy")
    # test_mlp()

    mlp_torch = MLPTorch(INPUT_SIZE, 32, 12, OUTPUT_SIZE)
    mlp_torch.load("../../../models/bipedalwalker/09-14-2019_23-38_NN=MLPTorchIndividal_POPSIZE=100_GEN"
                   "=1000_PMUTATION_0.3_PCROSSOVER_0.9.npy")
    # test_mlp_torch()

    env.close()

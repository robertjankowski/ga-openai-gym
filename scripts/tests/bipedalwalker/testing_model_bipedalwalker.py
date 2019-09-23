import gym
import numpy as np
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


def test_mlp_torch(is_reduced=False):
    global observation
    for _ in range(500):
        env.render()
        observation = torch.from_numpy(observation).float()
        if is_reduced:
            observation = observation[:10]
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
    # env = gym.wrappers.Monitor(env, 'cartpole', video_callable=lambda episode_id: True, force=True)
    observation = env.reset()

    INPUT_SIZE = 24
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4

    rnn = RNN(10, 24, 12, OUTPUT_SIZE)
    rnn.load('../../../models/bipedalwalker/09-23-2019_07-09_NN=RNNIndividual_POPSIZE=50_GEN=2000_PMUTATION_0'
             '.3_PCROSSOVER_0.8.npy')
    # test_rnn()

    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # mlp.load("../../../models/bipedalwalker/09-07-2019_16-34-56_POPSIZE=30_GEN=5_MUTATION_0.609-07-2019_16-34"
    #          "-56_POPSIZE=30_GEN=5_MUTATION_0.6.npy")
    # test_mlp()

    # Model 09-14-2019 NN: 24 - 32 - 12 - 4
    # Model 90-21-2019 NN: 10 - 24 - 12 - 4
    mlp_torch = MLPTorch(10, 24, 12, OUTPUT_SIZE)
    mlp_torch.load("../../../models/bipedalwalker/09-21-2019_22-29_NN=MLPTorchIndividal_POPSIZE=50_GEN"
                   "=2000_PMUTATION_0.3_PCROSSOVER_0.8.npy")
    test_mlp_torch(is_reduced=True)

    env.close()

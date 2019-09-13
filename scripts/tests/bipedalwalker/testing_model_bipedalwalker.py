import gym
import torch

from nn.mlp import MLP
from nn.rnn import RNN


def test_mlp():
    global observation
    for _ in range(300):
        env.render()
        action = mlp.forward(observation)
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
        observation, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    observation = env.reset()

    INPUT_SIZE = 24
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4

    rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    rnn.load('../../../models/bipedalwalker/09-09-2019_18-05_NN=RNNIndividual_POPSIZE=30_GEN=40_PMUTATION_0'
             '.7_PCROSSOVER_0.9.npy')
    # test_rnn()

    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    mlp.load("../../../models/bipedalwalker/09-07-2019_16-34-56_POPSIZE=30_GEN=5_MUTATION_0.609-07-2019_16-34"
             "-56_POPSIZE=30_GEN=5_MUTATION_0.6.npy")
    # test_mlp()

    # Not working correctly
    model = torch.load("../../../models/bipedalwalker/09-06-2019_14-40-56_POPSIZE=1000_GEN=1000_MUTATION_0.6.pt")
    model.eval()
    for _ in range(300):
        env.render()
        observation = torch.from_numpy(observation).float()
        action = model(observation)
        action = action.detach().numpy()
        observation, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

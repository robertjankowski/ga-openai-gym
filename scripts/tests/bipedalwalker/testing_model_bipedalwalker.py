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


def test_mlp_torch(input_size=None, is_reduced=False):
    global observation
    for _ in range(2000):
        env.render()
        observation = torch.from_numpy(observation).float()
        if is_reduced:
            observation = observation[:input_size]
        action = mlp_torch.forward(observation)
        action = action.detach().numpy()
        observation, reward, done, _ = env.step(action)
        if done:
            break


def test_rnn(is_reduced=False):
    global observation
    hidden = rnn.init_hidden()
    for _ in range(300):
        env.render()
        observation = torch.from_numpy(observation).float()
        if is_reduced:
            observation = observation[:10]
        action, hidden = rnn.forward(observation, hidden)
        action = action.detach().numpy()
        # action = np.nan_to_num(action)
        observation, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    # env = gym.wrappers.Monitor(env, 'bipedalwalker', video_callable=lambda episode_id: True, force=True)
    env.seed(123)
    observation = env.reset()

    INPUT_SIZE = 24
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4

    rnn = RNN(10, 24, 12, OUTPUT_SIZE)
    # rnn.load('../../../models/bipedalwalker/09-23-2019_07-09_NN=RNNIndividual_POPSIZE=50_GEN=2000_PMUTATION_0'
    #        '.3_PCROSSOVER_0.8.npy')
    # test_rnn()

    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # mlp.load("../../../models/bipedalwalker/09-07-2019_16-34-56_POPSIZE=30_GEN=5_MUTATION_0.609-07-2019_16-34"
    #          "-56_POPSIZE=30_GEN=5_MUTATION_0.6.npy")
    # test_mlp()

    # Model 09-14-2019 NN: 24 - 32 - 12 - 4
    # Model 09-21-2019 NN: 10 - 24 - 12 - 4
    # Model 09-26-2019 NN: 5  - 16 - 12 - 4
    # Model 09-30-2019 NN: 10 - 16 - 12 - 4
    mlp_torch = MLPTorch(10, 16, 12, OUTPUT_SIZE)
    mlp_torch.load("../../../models/bipedalwalker/10-21-2019_02-57_NN=MLPTorchIndividual_POPSIZE=30_GEN"
                   "=6000_PMUTATION_0.6_PCROSSOVER_0.85_I=2432_SCORE=135.05759182155143.npy")
    test_mlp_torch(input_size=10, is_reduced=True)

    env.close()

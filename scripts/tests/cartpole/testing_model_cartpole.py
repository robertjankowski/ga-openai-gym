import gym

from nn.mlp import MLP


def test_cartpole(nn, file):
    global observation
    nn.load(file)
    for _ in range(500):
        env.render()
        action = nn.forward(observation)
        observation, reward, done, info = env.step(round(action.item()))
        if done:
            break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation = env.reset()

    nn = MLP(4, 2, 1)
    test_cartpole(nn, '../../../models/cartpole/09-09-2019_17-37_POPSIZE=100_GEN=20_PMUTATION_0.4_PCROSSOVER_0.9.npy')
    env.close()

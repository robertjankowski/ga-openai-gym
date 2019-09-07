import gym
from nn.neural_network import NeuralNetwork

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    observation = env.reset()

    nn = NeuralNetwork(24, 16, 4)
    nn.load(
        '../models/bipedalwalker/09-07-2019_16-34-56_POPSIZE=30_GEN=5_MUTATION_0.609-07-2019_16-34-56_POPSIZE=30_GEN'
        '=5_MUTATION_0.6.npy')

    for _ in range(300):
        env.render()
        action = nn.forward(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()

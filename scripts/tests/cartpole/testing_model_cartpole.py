import gym.wrappers

from nn.mlp import MLP
import pickle


def test_cartpole(nn, file):
    global observation
    nn.load(file)
    for _ in range(100):
        env.render()
        action = nn.forward(observation)
        observation, reward, done, info = env.step(round(action.item()))
        if done:
            break


def save_model(nn, filename):
    with open(filename, 'wb') as output:
        pickle.dump(nn, output)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = gym.wrappers.Monitor(env, 'cartpole', video_callable=lambda episode_id: True, force=True)
    observation = env.reset()

    nn = MLP(4, 2, 1)
    test_cartpole(nn, '../../../models/cartpole/09-09-2019_17-37_POPSIZE=100_GEN=20_PMUTATION_0.4_PCROSSOVER_0.9.npy')

    save_model(nn, "09-09-2019_17-37_POPSIZE=100_GEN=20_PMUTATION_0.4_PCROSSOVER_0.9.pkl")

    env.close()

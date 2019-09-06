import gym
import torch

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation = env.reset()

    model = torch.load('../models/cartpole/09-06-2019_14-13-25_POPSIZE=100_GEN=20_MUTATION_0.6.pt')
    model.eval()

    for _ in range(500):
        env.render()
        observation = torch.from_numpy(observation).float()
        action = model(observation)
        action = action.detach().numpy().item()
        observation, reward, done, info = env.step(round(action))
        if done:
            break

    env.close()

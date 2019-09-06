import gym
import torch

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    observation = env.reset()

    model = torch.load('../models/bipedalwalker/09-06-2019_13-42-49.pt')
    model.eval()

    for _ in range(300):
        env.render()
        observation = torch.from_numpy(observation).float()
        action = model(observation)
        action = action.detach().numpy()
        observation, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        if done:
            break
    env.close()

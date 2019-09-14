import gym
import numpy as np
import torch

from nn.conv import ConvNet

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    nn = ConvNet()

    for _ in range(100):
        env.render()
        obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
        obs = obs.reshape((-1, 3, 96, 96))
        action = nn.forward(obs)
        action = action.detach().numpy()[0]
        obs, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

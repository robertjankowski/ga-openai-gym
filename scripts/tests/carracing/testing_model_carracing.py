import gym
import numpy as np
import torch

from nn.conv import ConvNet

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    nn = ConvNet()
    nn.load("../../../models/carracing/09-26-2019_16-44_NN=ConvNetTorchIndividal_POPSIZE=100_GEN=1000_PMUTATION_0"
            ".2_PCROSSOVER_0.8.npy")

    for _ in range(100):
        env.render()
        obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
        obs = obs.reshape((-1, 3, 96, 96))
        action = nn.forward(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

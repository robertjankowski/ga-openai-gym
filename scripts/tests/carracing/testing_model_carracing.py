import gym
import numpy as np
import torch

from nn.conv import ConvNet

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env.seed(112)
    env = gym.wrappers.Monitor(env, 'carracing_video', video_callable=lambda episode_id: True, force=True)

    obs = env.reset()
    nn = ConvNet()
    nn.load("../../../models/carracing/12-24-2019_22-50_NN=ConvNetTorchIndividal_POPSIZE=50_GEN=2000_PMUTATION_0"
            ".4_PCROSSOVER_0.8_I=1902_SCORE=440.54054054053455.npy")

    for _ in range(10000):
        env.render()
        obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
        obs = obs.reshape((-1, 3, 96, 96))
        action = nn.forward(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        if done:
            break

    #torch.save(nn, "12-17-2019_12-14_NN=ConvNetTorchIndividal_POPSIZE=50_GEN=2000_PMUTATION_0"
    #        ".4_PCROSSOVER_0.8_I=1495_SCORE=426.3157894736778.pt")

    env.close()

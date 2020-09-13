import slimevolleygym
import gym
import numpy as np
import torch
from gym import wrappers

# States:
# 12-dim vector

# Actions:
# 3-dim vector
#
# action[0] > 0 -> move forward
# action[1] > 0 -> move backward
# action[2] > 0 -> jump
from nn.mlp import DeepMLPTorch


def random_policy():
    return np.random.choice([-1, 1], size=3)


def mlp_policy(env, model, input_size):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs = torch.from_numpy(obs).float()
        obs = obs[:input_size]
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)


def main():
    env = gym.make('SlimeVolley-v0')
    # env = gym.wrappers.Monitor(env, 'slimevolley_119', video_callable=lambda episode_id: True, force=True)
    env.seed(119)
    input_size = 12
    output_size = 3
    hidden_sizes = [20, 20]
    model = DeepMLPTorch(input_size, output_size, *hidden_sizes)
    model.load_state_dict(
        torch.load(
            '../../../models/slimevolleyball/model-layers=12-[20, 20]-308-16-2020_18-22_NN=DeepMLPTorchIndividual_POPSIZE=30_GEN=2000_PMUTATION_0.1_PCROSSOVER_0.8_I=1824_SCORE=203.4210000000015.npy')
    )
    mlp_policy(env, model, input_size)


if __name__ == '__main__':
    main()

import gym
from gym import spaces
import numpy as np

env = gym.make("BipedalWalkerHardcore-v2")
env.reset()
print(env.action_space)  # Valid actions
print(env.action_space.low)
print(env.action_space.high)

space = spaces.Box(low=np.array([0, -1, -1, 1]), high=np.array([0, 1, 1, 0]))

for _ in range(300):
    env.render()
    observation, reward, done, info = env.step(space.sample())
    print(f"Reward: {reward}")
    if done:
        break
env.close()

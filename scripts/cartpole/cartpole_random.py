import numpy as np
import gym

from util.timing import timing


def run_episode(env, strategy, episode_len=100, render=False) -> int:
    """
    :param env:
    :param strategy: Either push cart to the left or right
    :param episode_len:
    :param render: Display cart
    :return: Total reward after one episode
    """
    total_reward = 0
    env.reset()
    for t in range(episode_len):
        if render:
            env.render()
        obs, reward, done, _ = env.step(strategy)
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate_strategy(env, strategy, n_episodes=200) -> float:
    """
    :param env:
    :param strategy: Either push cart to the left or right
    :param n_episodes:
    :return: Average total rewards
    """
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, strategy)
    return total_rewards / n_episodes


@timing
def random_actions(env, n_strategy=1000) -> None:
    """
        Actions - Discrete(2,)
         0 -> Push cart to the left
         1 -> Push cart to the right
        :return: 0 or 1
    """
    get_random_action = lambda x: np.random.randint(x)
    strategy_set = [get_random_action(2) for _ in range(n_strategy)]
    strategy_score = [evaluate_strategy(env, s) for s in strategy_set]
    print(f"Best score {np.max(strategy_score)}")


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.seed(123)
    POPULATION_SIZE = 100
    random_actions(env)
    env.close()

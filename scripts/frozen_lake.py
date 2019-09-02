import gym
import numpy as np
import time


def get_random_policy() -> np.array:
    # 4 possible moves (up, down, left, right)
    # 4x4 grid => 16 values
    return np.random.choice(4, size=16)


def run_episode(env, policy, episode_len=100, render=False) -> float:
    total_reward = 0
    obs = env.reset()
    for t in range(episode_len):
        if render:
            env.render()
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, n_episodes=100) -> float:
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes


def crossover(policy1, policy2, p=0.8) -> np.array:
    new_policy = policy1.copy()
    for i in range(16):
        if np.random.uniform() < p:
            new_policy[i] = policy2[i]
    return new_policy


def mutation(policy, p=0.02) -> np.array:
    new_policy = policy.copy()
    for i in range(16):
        if np.random.uniform() < p:
            new_policy[i] = np.random.choice(4)
    return new_policy


if __name__ == '__main__':
    np.random.seed(123)
    env = gym.make('FrozenLake-v0')
    env.seed(123)

    n_policy = 100
    n_steps = 20
    start = time.time()
    policy_pop = [get_random_policy() for _ in range(n_policy)]
    for idx in range(n_steps):
        policy_score = [evaluate_policy(env, p) for p in policy_pop]
        print(f'Generation: {idx}, max score: {np.max(policy_score)}')
        policy_ranks = np.flip((np.argsort(policy_score)))
        elite_set = [policy_pop[x] for x in policy_ranks[:5]]  # first 5 best individual
        select_probs = policy_score / np.sum(policy_score)
        child_set = [crossover(
            policy_pop[np.random.choice(range(n_policy), p=select_probs)],
            policy_pop[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - 5)]
        mutated_list = [mutation(p) for p in child_set]
        policy_pop = elite_set
        policy_pop += mutated_list
    policy_score = [evaluate_policy(env, p) for p in policy_pop]
    best_policy = policy_pop[np.argmax(policy_score)]
    end = time.time()

    print(f"Best score {np.max(policy_score)}, Total time: {end - start}s")

    # Evaluation
    env.reset()
    for _ in range(10):
        run_episode(env, best_policy, render=True)
    env.close()

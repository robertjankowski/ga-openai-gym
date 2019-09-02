import gym
import numpy as np
import torch
from typing import Tuple, Any
from collections import OrderedDict
import torch.distributions as tdist

from util.timing import timing


# Episode Termination
#  1. Pole Angle is more than ±12°
#  2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
#  3. Episode length is greater than 200

# Reward is 1 for every step taken, including the termination step

# Observation - Box(4,)
# 0 -> Cart Position        < -2.4, 2.4 >
# 1 -> Cart Velocity        < -Inf, Inf >
# 2 -> Pole Angle           < ~ -41.8°, ~ 41.8° >
# 3 -> Pole Velocity at Tip < -Inf, Inf >


def get_random_action() -> np.array:
    """
    Actions - Discrete(2,)
     0 -> Push cart to the left
     1 -> Push cart to the right
    :return: 0 or 1
    """
    return np.random.randint(2)


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
    strategy_set = [get_random_action() for _ in range(n_strategy)]
    strategy_score = [evaluate_strategy(env, s) for s in strategy_set]
    print(f"Best score {np.max(strategy_score)}")


def run_single(model, n_episodes=200, render=False) -> Tuple[int, Any]:
    """
    Calculate fitness function for given model

    :param model:
    :param n_episodes:
    :param render:
    :return: total_reward, models parameters (weights and biases) as list
    """
    obs = env.reset()
    total_reward = 0
    for _ in range(n_episodes):
        if render:
            env.render()
        obs = torch.from_numpy(obs).float()
        action = model(obs)
        action = action.detach().numpy().astype(np.int).item()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    weight_biases = parameters_list_from_state_dict(model)
    return total_reward, weight_biases


def parameters_list_from_state_dict(model) -> torch.Tensor:
    """
    Get model parameters (weights and biases) and concatenate them

    :param model
    :return: tensor with all weights and biases
    """
    parameters = model.state_dict().values()
    parameters = [x.flatten() for x in parameters]
    parameters = torch.cat(parameters, 0)
    return parameters


def from_parameters_list_to_order_dict(model_dict, model_parameters) -> OrderedDict:
    """
    Transform list of model parameters to state dict

    :param model_dict: OrderedDict, model schema
    :param model_parameters: List of model parameters
    :return:
    """
    shapes = [x.shape for x in model_dict.values()]
    shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

    partial_split = model_parameters.split(shapes_prod)
    model_values = []
    for i in range(len(shapes)):
        model_values.append(partial_split[i].view(shapes[i]))

    state_dict = OrderedDict((key, value) for (key, value) in zip(model_dict.keys(), model_values))
    return state_dict


def crossover(parent1, parent2):
    position = np.random.randint(0, parent1.shape[0])
    child1 = parent1.clone()
    child2 = parent2.clone()

    tmp = child1[:position].clone()
    child1[:position] = child2[:position]
    child2[:position] = tmp
    return child1, child2


def mutation(parent, p=0.05):
    """
    Mutate parent using normal distribution

    :param parent:
    :param p: Mutation rate
    """
    child = parent.clone()
    if np.random.rand() < p:
        position = np.random.randint(0, parent.shape[0])
        n = tdist.Normal(child.mean(), child.std())
        child[position] = n.sample((1,))
    return child


def create_model(d_in, h, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d_out),
        torch.nn.Sigmoid()
    )


def generation(d_in, h, d_out, population_size=100):
    # TODO: how to select individual from population ?

    for t in range(population_size):
        model = create_model(d_in, h, d_out)
        fitness, parameters = run_single(model)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.seed(123)

    # H - size of hidden layer
    N, D_in, H, D_out = 1, 4, 2, 1
    POPULATION_SIZE = 2000
    total_reward = [run_single(create_model(D_in, H, D_out))[0] for _ in range(POPULATION_SIZE)]
    print(np.max(total_reward), np.min(total_reward), np.average(total_reward))
    random_actions(env)
    env.close()

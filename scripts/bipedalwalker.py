from collections import OrderedDict
from typing import Tuple, Any, List
from datetime import datetime

import copy
import gym
import numpy as np
import torch
import torch.distributions as tdist


class Individual:
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            # H - size of hidden layer
            D_in, H, D_out = 24, 16, 4
            self.model = create_model(D_in, H, D_out)
        self.fitness = 0.0
        self.weights_biases = None

    def calculate_fitness(self) -> None:
        self.fitness, self.weights_biases = run_single(self.model)

    def update_model(self):
        # Update model weights and biases
        self.model.load_state_dict(from_parameters_list_to_order_dict(self.model.state_dict(), self.weights_biases))


def create_model(d_in, h, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d_out),
        torch.nn.Sigmoid()
    )


def run_single(model, n_episodes=1000, render=False) -> Tuple[int, Any]:
    """
    Calculate fitness function for given model

    :param model:
    :param n_episodes:
    :param render:
    :return: fitness, models parameters (weights and biases) as list
    """
    obs = env.reset()
    fitness = 0
    for _ in range(n_episodes):
        if render:
            env.render()
        obs = torch.from_numpy(obs).float()
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            break
    weight_biases = parameters_list_from_state_dict(model)
    return fitness, weight_biases


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


def crossover(parent1_weights_biases, parent2_weights_biases):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = parent1_weights_biases.clone()
    child2_weights_biases = parent2_weights_biases.clone()

    tmp = child1_weights_biases[:position].clone()
    child1_weights_biases[:position] = child2_weights_biases[:position]
    child2_weights_biases[:position] = tmp
    return child1_weights_biases, child2_weights_biases


def mutation(parent_weights_biases, p=0.7):
    """
    Mutate parent using normal distribution

    :param parent_weights_biases:
    :param p: Mutation rate
    """
    child_weight_biases = parent_weights_biases.clone()
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        child_weight_biases[position] = np.random.randint(-20, 20)
    return child_weight_biases


def statistics(population: List[Individual]) -> Tuple[float, float, float]:
    population_fitness = list(map(lambda individual: individual.fitness, population))
    mean = np.mean(population_fitness)
    min = np.min(population_fitness)
    max = np.max(population_fitness)
    return mean, min, max


def selection(population) -> int:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[0], sorted_population[1]
    return parent1, parent2


def generation(old_population, new_population) -> List[Individual]:
    for i in range(0, len(old_population) - 1, 2):
        parent1, parent2 = selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases, parent2.weights_biases)

        # Mutation
        child1.weights_biases = mutation(child1.weights_biases)
        child2.weights_biases = mutation(child2.weights_biases)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness()
        child2.calculate_fitness()

        # If children fitness is greater thant parents update population
        new_population[i] = child1
        new_population[i + 1] = child2


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)

    POPULATION_SIZE = 100
    MAX_GENERATION = 200

    old_population = [Individual() for _ in range(POPULATION_SIZE)]
    new_population = [None] * POPULATION_SIZE

    for _ in range(MAX_GENERATION):
        [p.calculate_fitness() for p in old_population]
        generation(old_population, new_population)

        mean, min, max = statistics(new_population)
        old_population = copy.deepcopy(new_population)
        print(f"Mean: {mean}\tmin: {min}\tmax: {max}")

    best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]

    date = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    torch.save(best_model.model, '../models/bipedalwalker/{}.pt'.format(date))
    env.close()

import copy
from datetime import datetime
from typing import Tuple, List

import gym
import numpy as np

from util.timing import timing
from nn.neural_network import NeuralNetwork


class Individual:
    def __init__(self):
        D_in, H, D_out = 24, 16, 4
        self.nn = NeuralNetwork(D_in, H, D_out)
        self.fitness = 0.0
        self.weights_biases = self.nn.get_weights_biases()

    def calculate_fitness(self, env) -> None:
        self.fitness, self.weights_biases = run_single(env, self.nn)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)


def run_single(env, model, n_episodes=1000, render=False) -> Tuple[int, np.array]:
    """
    Calculate fitness function for given model

    :param env:
    :param model:
    :param n_episodes:
    :param render:
    :return: fitness
    """
    obs = env.reset()
    fitness = 0
    for _ in range(n_episodes):
        if render:
            env.render()
        action = model.forward(obs)
        obs, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            break
    return fitness, model.get_weights_biases()


def crossover(parent1_weights_biases, parent2_weights_biases):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    child1_weights_biases[:position], child2_weights_biases[:position] = \
        child2_weights_biases[:position], child1_weights_biases[:position]
    return child1_weights_biases, child2_weights_biases


def mutation(parent_weights_biases, p=0.6):
    """
    Mutate parent using normal distribution

    :param parent_weights_biases:
    :param p: Mutation rate
    """
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = 5 * n + np.random.randint(-20, 20)
    return child_weight_biases


def selection(population) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[0], sorted_population[1]
    return parent1, parent2


def generation(env, old_population, new_population, p_mutation) -> List[Individual]:
    for i in range(0, len(old_population) - 1, 2):
        parent1, parent2 = selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases, parent2.weights_biases)

        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)

        # If children fitness is greater thant parents update population
        new_population[i] = child1
        new_population[i + 1] = child2


def statistics(population: List[Individual]) -> Tuple[float, float, float]:
    population_fitness = list(map(lambda individual: individual.fitness, population))
    mean = np.mean(population_fitness)
    min = np.min(population_fitness)
    max = np.max(population_fitness)
    return mean, min, max


def update_population_fitness(env, population):
    for p in population:
        p.calculate_fitness(env)


@timing
def main(env, pop_size, max_generation, p_mutation, log=False):
    date = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    path = '{}_POPSIZE={}_GEN={}_MUTATION_{}'.format(date, pop_size,
                                                     max_generation,
                                                     p_mutation)

    old_population = [Individual() for _ in range(pop_size)]
    new_population = [None] * pop_size

    for i in range(max_generation):
        update_population_fitness(env, old_population)
        generation(env, old_population, new_population, p_mutation)

        mean, min, max = statistics(new_population)
        old_population = copy.deepcopy(new_population)
        stats = f"Generation {i + 1} | Mean: {mean}\tmin: {min}\tmax: {max}\n"
        if log:
            with open(path + '.log', "a") as f:
                f.write(stats)
        print(stats)

    best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]

    model_path = '../models/bipedalwalker/' + path
    np.save(model_path + path + '.npy', best_model.nn.get_weights_biases())


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)

    POPULATION_SIZE = 30
    MAX_GENERATION = 5
    MUTATION_RATE = 0.6

    main(env, POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE)

    env.close()

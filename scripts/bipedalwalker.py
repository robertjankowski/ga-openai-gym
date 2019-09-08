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
        self.fitness = 100 + self.fitness

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)


def run_single(env, model, n_episodes=300, render=False) -> Tuple[int, np.array]:
    """
    Calculate fitness function for given model
    """
    obs = env.reset()
    fitness = 0
    for episode in range(n_episodes):
        if render:
            env.render()
        action = model.forward(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            fitness /= episode
            break
        fitness += reward
    return fitness, model.get_weights_biases()


def crossover(parent1_weights_biases, parent2_weights_biases, p_crossover):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p_crossover:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]

    return child1_weights_biases, child2_weights_biases


def mutation(parent_weights_biases, p):
    """
    Mutate parent using normal distribution

    :param parent_weights_biases:
    :param p: Mutation rate
    """
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = 5 * n  # + np.random.randint(-20, 20)
    return child_weight_biases


def ranking_selection(population) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[0], sorted_population[1]
    return parent1, parent2


def roulette_wheel_selection(population: List[Individual]):
    total_fitness = np.sum([individual.fitness for individual in population])
    selection_probabilities = [individual.fitness / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]


def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1 = roulette_wheel_selection(old_population)
        parent2 = roulette_wheel_selection(old_population)
        # parent1, parent2 = ranking_selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)

        # If children fitness is greater thant parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


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
def main(env, pop_size, max_generation, p_mutation, p_crossover, log=False):
    date = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    path = '{}_POPSIZE={}_GEN={}_MUTATION_{}'.format(date, pop_size,
                                                     max_generation,
                                                     p_mutation)

    old_population = [Individual() for _ in range(pop_size)]
    new_population = [None] * pop_size

    for i in range(max_generation):
        update_population_fitness(env, old_population)
        generation(env, old_population, new_population, p_mutation, p_crossover)

        mean, min, max = statistics(new_population)
        old_population = copy.deepcopy(new_population)
        stats = f"Generation {i + 1} | Mean: {mean}\tmin: {min}\tmax: {max}\n"
        if log:
            with open(path + '.log', "a") as f:
                f.write(stats)
        print(stats)

    best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]

    model_path = '../models/bipedalwalker/' + path
    np.save(model_path + '.npy', best_model.nn.get_weights_biases())


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)

    POPULATION_SIZE = 100
    MAX_GENERATION = 40
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    main(env, POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE)

    env.close()

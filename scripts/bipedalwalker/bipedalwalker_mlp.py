import copy
from typing import Tuple

import gym
import numpy as np
import torch

from ga.individual import crossover, mutation, Individual, ranking_selection, crossover_new, inversion, mutation_gen
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.mlp import MLP, MLPTorch


class MLPIndividual(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return MLP(input_size, hidden_size, output_size)

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = obs[:INPUT_SIZE]
            action = self.nn.forward(obs)
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


class MLPTorchIndividual(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return MLPTorch(input_size, hidden_size, 12, output_size)

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = obs[:INPUT_SIZE]
            obs = torch.from_numpy(obs).float()
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        # parent1 = roulette_wheel_selection(old_population)
        # parent2 = roulette_wheel_selection(old_population)
        parent1, parent2 = ranking_selection(old_population)

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


def generation_new(env, old_population, new_population, p_mutation, p_crossover, p_inversion):
    """
    1. Tournament selection to create new population 1
    2. Crossover to new population 2 (because sometimes it doesn't happen)
    3. Mutation of each gen (single value in genotype)
    4. Inversion of genotype
    """
    import random
    pop_size = len(old_population)
    new_pop = []

    # 1.
    for i in range(pop_size):
        indv1 = random.choice(old_population)
        indv2 = random.choice(old_population)
        new_pop.append(indv1 if indv1.fitness > indv2.fitness else indv2)

    # 2.
    while len(new_population) < pop_size:
        parent1 = random.choice(new_pop)
        parent2 = random.choice(new_pop)
        if np.random.rand() < p_crossover:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            child1.weights_biases, child2.weights_biases = crossover_new(parent1.weights_biases, parent2.weights_biases)

            child1.update_model()
            child2.update_model()
            child1.calculate_fitness(env)
            child2.calculate_fitness(env)

            # 3.
            mutation_gen(child1.weights_biases, p_mutation)
            mutation_gen(child2.weights_biases, p_mutation)

            # 4.
            if np.random.rand() < p_inversion:
                child1.weights_biases = inversion(child1.weights_biases)
                child2.weights_biases = inversion(child2.weights_biases)

            new_population.append(child1)
            new_population.append(child2)


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)

    POPULATION_SIZE = 10
    MAX_GENERATION = 20
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.8
    INVERSION_RATE = 0.02

    INPUT_SIZE = 10
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4

    p = Population(MLPTorchIndividual(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE),
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE,
                   CROSSOVER_RATE,
                   INVERSION_RATE)
    p.run(env, generation_new, verbose=True, log=True, output_folder='')

    env.close()

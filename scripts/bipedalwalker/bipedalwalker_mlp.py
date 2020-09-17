import copy
from typing import Tuple

import gym
import numpy as np
import torch

from ga.individual import crossover, mutation, Individual, ranking_selection
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.mlp import DeepMLPTorch

HIDDEN_SIZE = [10, 12]


class DeepBipedalWalkerIndividual(Individual):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.input_size = input_size

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return DeepMLPTorch(input_size, output_size, *HIDDEN_SIZE)

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = obs[:self.input_size]
            obs = torch.from_numpy(obs).float()
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(env,
               old_population: list,
               new_population: list,
               p_mutation: float,
               p_crossover: float,
               p_inversion: float = 0.0):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
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


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.seed(123)

    POPULATION_SIZE = 10
    MAX_GENERATION = 10
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.8

    INPUT_SIZE = 24
    OUTPUT_SIZE = 4

    assert POPULATION_SIZE % 2 == 0
    p = Population(DeepBipedalWalkerIndividual(INPUT_SIZE, 0, OUTPUT_SIZE),
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE,
                   CROSSOVER_RATE,
                   0.0)
    p.set_population([DeepBipedalWalkerIndividual(INPUT_SIZE, 0, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)])
    p.run(env, generation, verbose=True, log=True,
          output_folder=f'model-layers={INPUT_SIZE}-{HIDDEN_SIZE}-{OUTPUT_SIZE}', save_as_pytorch=False)
    env.close()

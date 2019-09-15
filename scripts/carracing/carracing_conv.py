import copy
from typing import Tuple

import gym
import numpy as np
import torch

from ga.individual import ranking_selection, crossover, mutation, Individual
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.conv import ConvNet


class ConvNetTorchIndividal(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return ConvNet()

    def run_single(self, env, n_episodes=100, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
            obs = obs.reshape((-1, 3, 96, 96))
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


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env.seed(123)

    POPULATION_SIZE = 100
    MAX_GENERATION = 1000
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    p = Population(ConvNetTorchIndividal(None, None, None), POPULATION_SIZE, MAX_GENERATION,
                   MUTATION_RATE, CROSSOVER_RATE)
    p.run(env, generation, verbose=True, output_folder='')

    env.close()

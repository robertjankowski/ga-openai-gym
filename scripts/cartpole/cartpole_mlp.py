import copy
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple

import gym
import numpy as np
import torch

from ga.individual import roulette_wheel_selection, crossover, mutation, Individual

# Observation - Box(4,)
# 0 -> Cart Position        < -2.4, 2.4 >
# 1 -> Cart Velocity        < -Inf, Inf >
# 2 -> Pole Angle           < ~ -41.8°, ~ 41.8° >
# 3 -> Pole Velocity at Tip < -Inf, Inf >


# > Episode Termination
#    1. Pole Angle is more than ±12°
#    2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
#    3. Episode length is greater than 200
# > Reward is 1 for every step taken, including the termination step


# class Individual:
#     def __init__(self, model=None):
#         if model is not None:
#             self.model = model
#         else:
#             # H - size of hidden layer
#             D_in, H, D_out = 4, 2, 1
#             self.model = create_model(D_in, H, D_out)
#         self.fitness = 0.0
#         self.weights_biases = None
#
#     def calculate_fitness(self) -> None:
#         self.fitness, self.weights_biases = run_single(self.model)
#
#     def update_model(self):
#         # Update model weights and biases
#         self.model.load_state_dict(from_parameters_list_to_order_dict(self.model.state_dict(), self.weights_biases))
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.mlp import MLP


class MLPIndividual(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return MLP(input_size, hidden_size, output_size)

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for _ in range(n_episodes):
            if render:
                env.render()
            action = self.nn.forward(obs)
            obs, reward, done, _ = env.step(round(action.item()))
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1 = roulette_wheel_selection(old_population)
        parent2 = roulette_wheel_selection(old_population)

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
    env = gym.make('CartPole-v1')
    env.seed(123)

    POPULATION_SIZE = 100
    MAX_GENERATION = 20

    MUTATION_RATE = 0.4
    CROSSOVER_RATE = 0.9

    INPUT_SIZE = 4
    HIDDEN_SIZE = 2
    OUTPUT_SIZE = 1

    p = Population(MLPIndividual(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE),
                   POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE)
    p.run(env, generation, verbose=True, output_folder='../../models/cartpole')

    env.close()

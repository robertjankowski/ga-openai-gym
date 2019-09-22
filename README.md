## Using genetic algorithms to train neural network in OpenAI gym environment. 
***

 | Random BipedalWalker-v2 | GA BipedalWalker-v2 | 
| ------------- |:-------------:|
| ![mlp_bipedal](docs/bipedalwalker/bipedalwalker_random.gif) | ![mlp_bipedal](docs/bipedalwalker/mlp_bipedalwalker.gif) |

 | Random Cartpole-v0 | GA Cartpole-v0 | 
| ------------- |:-------------:|
| ![cartpole-random](docs/cartpole/cartpole_random.gif) | ![cartpole-random](docs/cartpole/cartpole_nn.gif) | $1600 |


### Explanation

##### Overview
From Cartpole-v0 we get vector of observations and reward for each action (0/1) given to the environment.
The observations is feed into MLP with 4-2-1 architecture and sigmoid function as the output.
Usually to train the neural network we can use backpropagation algorithm, however here I use a genetic algorithm.
The goal of the GA is to find the optimal values of weights and biases for a given network. 

##### Fitness function

Function fitness is the sum of rewards. Here is sample code (`nn` is a single neural network):
```python
env = gym.make('CartPole-v0')

def get_fitness(n_episodes):
    obs = env.reset()
    fitness = 0
    for episode in range(n_episodes):
        action = nn.forward(obs)
        obs, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            break
    return fitness
```

##### Schema

1. Create initial population of neural networks. At first the weights and biases are randomly initialized.
2. Calculate fitness function for each individual.
3. Weights and biases are transformed into a single vector.
4. Selection (elitism selection or roulette wheel selection)
5. Crossover 
6. Mutation
7. If child fitness is greater than parent fitness score the new population is updated
8. Go to 2.

| ![mlp-cartpole](docs/bipedalwalker/mlp.png) |
|:---:| 
| Architecture of neural network used in BipedalWalker-v2 problem |

## [BipedalWalker-v2](https://github.com/openai/gym/wiki/BipedalWalker-v2)

**Environment** (continuous)
 - 24 observations (hull_angle, vel_x, vel_y and many more)

**Actions** (continues):
 - Hip_1 and Hip_2 (Torque / Velocity)
 - Knee_1 and Knee_2 (Torque / Velocity)
 
**Reward** is given for moving forward, total 300+ points up to the far end. 
If the robot falls, it gets -100

**Episode Termination**
The episode ends when the robot body touches ground or the robot reaches far right side of the environment

**Neural network architecture**: simple MLP
 - Input size: 10
 - One hidden layer, size: 24
 - Second hidden layer, size: 12
 - Output layer with `tanh` activation function, size: 4
 
**Genetic algorithm parameters**:
 - population size: 50
 - generation: 2000
 - mutation rate: 0.3
 - crossover rate: 0.9


## [Cartpole-v0](https://github.com/openai/gym/wiki/CartPole-v0)

**Environment** (continuous)
 - 4 observations (Cart Position and Velocity, Pole Angle and Velocity at Tip)

**Actions** (discrete):
 - Push cart to the left or right

**Reward**: 1 for every step taken, including the termination step

**Episode Termination**
 - Pole Angle is more than ±12°
 - Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
 - Episode length is greater than 200

**Neural network architecture**: simple MLP
 - Input size: 4
 - One hidden layer, size: 2
 - Output layer with Softmax activation function, size: 1
 
**Genetic algorithm parameters**:
 - population size: 100
 - generation: 20
 - mutation rate: 0.4
 - crossover rate: 0.9
 
***

Run on [**Spell**](https://web.spell.run/robjan/)
```shell script
spell run "python scripts/bipedalwalker.py" --pip-req requirements.txt
```
For car racing (need to open virual display server)
```shell script
spell run --apt python-dev --apt cmake --apt zlib1g-dev --apt libjpeg-dev \
	--apt xvfb --apt ffmpeg --apt xorg-dev --apt python-opengl --apt libboost-all-dev \
	--apt libsdl2-dev --apt swig \ 
	"Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log ; export DISPLAY=:1 ; python scripts/spell/carracing_conv_spell.py" \ 
	--pip-req requirements.txt
```

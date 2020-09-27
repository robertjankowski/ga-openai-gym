import gym
import tensorflow as tf
import numpy as np

from scripts.multitask_learning.multitask_learning import MultiTaskModel, Shape

from gym import wrappers

BIPEDALWALKER_RANDOM = 100
CARTPOLE_RANDOM = 1


def convert_actions(action_cartpole, action_bipedalwalker):
    action_cartpole = round(action_cartpole.numpy().item())
    if action_cartpole > 1:
        action_cartpole = 1
    if action_cartpole < 0:
        action_cartpole = 0
    action_bipedalwalker = action_bipedalwalker.numpy()[0]
    return action_cartpole, action_bipedalwalker


def both_correct_obs(model, obs_cartpole, obs_bipedalwalker, bipedalwalker_shape, cartpole_shape):
    action_cartpole, action_bipedalwalker = model({
        "first_input": np.reshape(obs_cartpole, (1, cartpole_shape.input)),
        "second_input": np.reshape(obs_bipedalwalker, (1, bipedalwalker_shape.input))
    })
    return convert_actions(action_cartpole, action_bipedalwalker)


def first_noise_second_correct_obs(model, obs_cartpole, obs_bipedalwalker, bipedalwalker_shape, cartpole_shape):
    action_cartpole, action_bipedalwalker = model({
        "first_input": np.reshape(np.random.uniform(-CARTPOLE_RANDOM, CARTPOLE_RANDOM, cartpole_shape.input),
                                  (1, cartpole_shape.input)),
        "second_input": np.reshape(obs_bipedalwalker, (1, bipedalwalker_shape.input))
    })
    return convert_actions(action_cartpole, action_bipedalwalker)


def first_correct_second_noise_obs(model, obs_cartpole, obs_bipedalwalker, bipedalwalker_shape, cartpole_shape):
    action_cartpole, action_bipedalwalker = model({
        "first_input": np.reshape(obs_cartpole, (1, cartpole_shape.input)),
        "second_input": np.reshape(
            np.random.uniform(-BIPEDALWALKER_RANDOM, BIPEDALWALKER_RANDOM, bipedalwalker_shape.input),
            (1, bipedalwalker_shape.input))
    })
    return convert_actions(action_cartpole, action_bipedalwalker)


def both_noise_obs(model, obs_cartpole, obs_bipedalwalker, bipedalwalker_shape, cartpole_shape):
    action_cartpole, action_bipedalwalker = model({
        "first_input": np.reshape(np.random.uniform(-CARTPOLE_RANDOM, CARTPOLE_RANDOM, cartpole_shape.input),
                                  (1, cartpole_shape.input)),
        "second_input": np.reshape(
            np.random.uniform(-BIPEDALWALKER_RANDOM, BIPEDALWALKER_RANDOM, bipedalwalker_shape.input),
            (1, bipedalwalker_shape.input))
    })
    return convert_actions(action_cartpole, action_bipedalwalker)


ACTION_TYPE = {
    "both_correct_obs": both_correct_obs,
    "first_noise_second_correct_obs": first_noise_second_correct_obs,
    "first_correct_second_noise_obs": first_correct_second_noise_obs,
    "both_noise_obs": both_noise_obs
}


def test_model(env_bipedalwalker, env_cartpole, model, bipedalwalker_shape, cartpole_shape, action_type: str):
    obs_bipedalwalker = env_bipedalwalker.reset()
    obs_cartpole = env_cartpole.reset()
    i, counter = 0, 0
    while i < 2000:
        # env_bipedalwalker.render()
        # env_cartpole.render()
        action_cartpole, action_bipedalwalker = ACTION_TYPE[action_type](model, obs_cartpole, obs_bipedalwalker,
                                                                         bipedalwalker_shape, cartpole_shape)
        obs_bipedalwalker, _, done_bipedalwalker, _ = env_bipedalwalker.step(action_bipedalwalker)
        obs_cartpole, _, done_cartpole, _ = env_cartpole.step(action_cartpole)

        if done_bipedalwalker or done_cartpole:
            counter += 1
        if counter > 200:
            break
        i += 1


if __name__ == '__main__':
    obs_type = "both_noise_obs"

    env_bipedalwalker = gym.make('BipedalWalker-v2')
    env_cartpole = gym.make("CartPole-v0")
    env_bipedalwalker = gym.wrappers.Monitor(env_bipedalwalker, obs_type + "_bipedalwalker",
                                             video_callable=lambda episode_id: True, force=True)
    env_cartpole = gym.wrappers.Monitor(env_cartpole, obs_type + "_cartpole",
                                        video_callable=lambda episode_id: True, force=True)

    seed = 123
    env_bipedalwalker.seed(seed)
    env_cartpole.seed(seed)

    cartpole_shape = Shape(4, 1)
    bipedalwalker_shape = Shape(24, 4)
    hidden_sizes = [12, 20, 4]
    model = MultiTaskModel(cartpole_shape, bipedalwalker_shape, *hidden_sizes)
    model_path = tf.train.latest_checkpoint("../../../models/multitask-model-test")
    model.load_weights(model_path)

    test_model(env_bipedalwalker,
               env_cartpole,
               model.model,
               bipedalwalker_shape,
               cartpole_shape,
               obs_type)

    env_bipedalwalker.close()
    env_cartpole.close()

import gym
from nn.mlp import MLP


def single_run(env, model, render=False):
    obs = env.reset()
    features = []
    labels = []
    while True:
        if render:
            env.render()
        features.append(obs)
        action = model.forward(obs)
        labels.append(action)
        obs, reward, done, _ = env.step(round(action.item()))
        if done:
            break
    assert len(features) == len(labels)
    return features, labels


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    N_RUNS = 500
    model = MLP(4, 2, 1)
    model_path = "../../models/cartpole/cartpole12-27-2019_20-29_NN=MLPIndividual_POPSIZE=100_GEN=20_PMUTATION_0.4_PCROSSOVER_0.9.npy"
    model.load(model_path)

    output_file = "data_" + model_path.split('/')[-1].split('.')[0] + f"_NRUNS={N_RUNS}.csv"
    features_output_file = "features_" + output_file
    labels_output_file = "labels_" + output_file

    with open(features_output_file, 'w') as f_features, open(labels_output_file, 'w') as f_labels:
        for i in range(N_RUNS):
            env.seed(i)
            features, labels = single_run(env, model)
            for f, l in zip(features, labels):
                f_features.write(','.join(map(str, f)) + '\n')
                f_labels.write(','.join(map(str, l)) + '\n')
    env.close()

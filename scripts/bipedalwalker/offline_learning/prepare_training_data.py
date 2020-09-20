import gym
import torch

from nn.mlp import DeepMLPTorch


def single_run(env, model, render=False):
    obs = env.reset()
    features = []
    labels = []
    while True:
        if render:
            env.render()
        features.append(obs)
        obs = torch.from_numpy(obs).float()
        action = model.forward(obs).detach().numpy()
        labels.append(action)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    assert len(features) == len(labels)
    return features, labels


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    N_RUNS = 20
    INPUT_SIZE = 24
    HIDDEN_SIZE = [20, 12, 12]
    OUTPUT_SIZE = 4

    model_path = "../../../models/bipedalwalker/large_model/model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_PMUTATION_0.1_PCROSSOVER_0.85_I=4949_SCORE=53.64846821325179.npy"
    model = DeepMLPTorch(INPUT_SIZE, OUTPUT_SIZE, *HIDDEN_SIZE)
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

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from nn.mlp import DeepMLPTorch

INPUT_SHAPE = 8
OUTPUT_SHAPE = 4
HIDDEN_SIZES = [8]


def load_dataset(features_path: str, labels_path: str, use_columns=None):
    X = pd.read_csv(features_path, header=None, usecols=use_columns)
    y = pd.read_csv(labels_path, header=None)
    X = torch.tensor(X.values, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)
    return X, y


def create_model(input_size, output_size, hidden_size):
    model = DeepMLPTorch(input_size, output_size, hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


def train(train_data, model, opt, loss_func, epochs):
    min_loss = 999999
    for epoch in range(epochs):
        for xb, yb in train_data:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        new_min_loss = loss_func(student_model(xb), yb)
        if new_min_loss < min_loss:
            min_loss = new_min_loss
            torch.save(student_model,
                       "model-layers={}-[{}]-{}-NN=StudentModel-EPOCH={}_LOSS={}.pt".format(
                           INPUT_SHAPE, ','.join(map(str, HIDDEN_SIZES)), OUTPUT_SHAPE, epochs, min_loss))
        print('Epoch {}: loss = {}'.format(epoch, min_loss))


if __name__ == '__main__':
    X, y = load_dataset(
        "features_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_PMUTATION_0_NRUNS=10000_subset.csv",
        "labels_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_PMUTATION_0_NRUNS=10000_subset.csv",
        use_columns=list(range(INPUT_SHAPE))
    )

    BATCH_SIZE = 64
    EPOCHS = 50
    train_ds = TensorDataset(X, y)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    student_model, opt = create_model(INPUT_SHAPE, OUTPUT_SHAPE, *HIDDEN_SIZES)
    loss_func = nn.modules.loss.MSELoss(size_average=False)

    train(train_dl, student_model, opt, loss_func, EPOCHS)

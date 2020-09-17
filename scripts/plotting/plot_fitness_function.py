import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd


def plot_fitness_function(data):
    plt.figure(figsize=(8, 6))
    plt.grid(alpha=0.3)
    data_mean_smoothed = gaussian_filter1d(data["mean"], sigma=2)
    plt.plot(data["generation"], data["mean"], alpha=0.25, color='orange')
    plt.plot(data["generation"], data_mean_smoothed, label="Mean value", color='orange')
    plt.ylabel('Fitness function', fontsize=13)
    plt.xlabel('Generation', fontsize=13)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.savefig('../../docs/bipedalwalker/bipedalwalker-24-20-12-12-4.png', bbox_inches='tight', dpi=300)
    # plt.show()


def load_logs(path: str):
    df = pd.read_csv(path, header=None)
    df.columns = ['date', 'generation', 'mean', 'min', 'max']
    return df


if __name__ == '__main__':
    path = "../../models/bipedalwalker/large_model/model-layers=24-[20, 12, 12]-4logs.csv"

    data = load_logs(path)
    plot_fitness_function(data)

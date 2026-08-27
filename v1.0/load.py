from main import NeuralNetwork
from main import ActF
import numpy as np
import matplotlib.pyplot as plt

def plot(nn, sample_range=(0, 2*np.pi)):
    sample_count = 100

    x_plot = np.linspace(sample_range[0], sample_range[1], sample_count)
    y_plot = [nn.forward(np.array([x]).reshape(-1,1))[0] for x in x_plot]

    plt.figure()
    plt.plot(x_plot, y_plot, color="red", linestyle="--")
    plt.title("prediction")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":   
    label = "sin3x"
    nn = NeuralNetwork.load(f"{label}.npz")
    print(nn)
    plot(nn)
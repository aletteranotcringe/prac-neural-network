import numpy as np
from main import NeuralNetwork
from main import Activator as af
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import keyboard
import threading

stop_training = False
def kb_listener():
    global stop_training
    keyboard.wait("q")
    stop_training = True
threading.Thread(target=kb_listener, daemon=True).start()

def save_all(nn_array):
    for nn in nn_array:
        print(f"Exporting {nn.label}")
        nn.save(f"{nn.label}.npz")

def parallel_training(nn_array, 
        target_function,
        sample_range = (0, 1),
        epochs = 1000,
        batch_size = 50,
        epoch_plot_step = 50):
    
    # Define unique colors
    colors = cm.get_cmap('plasma', len(nn_array))

    x_plot = np.linspace(sample_range[0], sample_range[1], 100).reshape(-1, 1)
    y_true = target_function(x_plot)

    for epoch in range(epochs):
        if stop_training:
            save_all(nn_array)
            break

        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            x_val = np.random.uniform(sample_range[0], sample_range[1])
            x_batch.append([x_val])
            y_batch.append([target_function(x_val)])

        x_batch = np.array(x_batch).transpose()
        y_batch = np.array(y_batch).transpose()

        for nn in nn_array:
            nn.backprop(x_batch, y_batch)

        if epoch % epoch_plot_step == 0:
            plt.clf()
            plt.title(f"Epoch {epoch}")
            plt.plot(x_plot, y_true, color="red")
            for idx, nn in enumerate(nn_array):
                y_pred = np.array([nn.forward(np.array([x]).reshape(-1,1))[0] for x in x_plot])
                plt.plot(x_plot, y_pred, color=colors(idx), linestyle="--", label=nn.label)
            plt.grid(True)
            plt.legend()
            plt.pause(0.01)

    plt.figure()
    plt.title("Final Prediction")
    plt.plot(x_plot, y_true, label="target", color="red")
    for idx, nn in enumerate(nn_array):
        y_pred = np.array([nn.forward(np.array([i]).reshape(-1,1))[0] for i in x_plot])
        plt.plot(x_plot, y_pred, label=nn.label, color=colors(idx), linestyle="--")
    plt.grid(True)
    plt.legend() 
    plt.show()
    plt.pause(0.01)

def change_plot_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        'axes.facecolor': '#050914',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'figure.facecolor': '#01030d',
        'figure.edgecolor': 'black',
        'grid.color': '#555555'
    })

if __name__ == "__main__":       
    def target_function(x):
        return np.sin(x / 3)
        
    change_plot_style()    
    nn_array = [
        NeuralNetwork(
            shape=[1, 16, 8, 1],
            label="lr=0.01",
            learning_rate=0.01,
            act=af.scaled_tanh,
            act_deriv=af.scaled_tanh_deriv
        ),
        NeuralNetwork(
            shape=[1, 16, 8, 1],
            label="lr=0.1",
            learning_rate=0.1,
            act=af.scaled_tanh,
            act_deriv=af.scaled_tanh_deriv
        )
    ]

    print("Press q to halt training and export")

    parallel_training(nn_array, 
                      target_function, 
                      (0, 6 * np.pi), 
                      epochs=5000, 
                      batch_size=3, 
                      epoch_plot_step=10)
    save_all(nn_array)

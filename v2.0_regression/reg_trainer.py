import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from modules.neuralnetwork import NeuralNetwork

class TrainingHalt(Exception):
    """Exception raised when training is manually halted."""
    pass

class RegTrainer:
    """
    Manages training of one or more neural networks on a target function.
    
    Args:
        models (list[NeuralNetwork]): List of models to train.
        target_fn (callable): Function to approximate.
        sample_range (tuple): Range from which to sample input data.
        batch_size (int): Number of samples per batch.
    """
    def __init__(self, models, target_fn, sample_range=(0, 1), batch_size=50):
        self.models = models
        self.target_fn = target_fn
        self.sample_range = sample_range
        self.batch_size = batch_size
        self.stop_flag = False

        self.colors = cm.get_cmap('plasma', len(models))
        self.x_plot = np.linspace(*sample_range, 100).reshape(-1, 1)
        self.y_true = target_fn(self.x_plot)

        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        """Listens for 'q' key to stop training."""
        import keyboard
        threading.Thread(target=lambda: keyboard.wait("q") or setattr(self, 'stop_flag', True), daemon=True).start()

    def _batch(self):
        """Generates a random training batch."""
        x = np.random.uniform(*self.sample_range, (self.batch_size, 1))
        return x.T, self.target_fn(x).T

    def train(self, epochs=1000, plot_interval=50):
        """
        Trains all models on the target function.
        
        Args:
            epochs (int): Number of training iterations.
            plot_interval (int): Frequency of visual feedback.
        """
        for epoch in range(epochs):
            if self.stop_flag:
                raise TrainingHalt()

            x_batch, y_batch = self._batch()
            for model in self.models:
                model.backprop(x_batch, y_batch)

            if epoch % plot_interval == 0:
                self._plot(epoch)

    def _plot(self, epoch):
        """Plots current model predictions against the target function."""
        plt.clf()
        plt.title(f"Epoch {epoch}")
        plt.plot(self.x_plot, self.y_true, label="Target", color="red")

        for idx, model in enumerate(self.models):
            y_pred = np.array([model.forward(x.reshape(-1, 1))[0] for x in self.x_plot])
            plt.plot(self.x_plot, y_pred, label=model.label, color=self.colors(idx), linestyle="--")

        plt.grid(True)
        plt.legend()
        plt.pause(0.01)

    def save_all(self):
        """Saves all models to disk."""
        for model in self.models:
            print(f"Saving {model.label}")
            model.save(f"{model.label}.npz")

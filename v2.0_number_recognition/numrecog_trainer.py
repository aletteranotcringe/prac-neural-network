import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from modules.neuralnetwork import NeuralNetwork
from modules.loss_fn import get_lossfn

class TrainingHalt(Exception):
    pass

class NumRecogTrainer:
    def __init__(self, model: NeuralNetwork, batch_size=50):
        self.model = model
        self.batch_size = batch_size
        self.stop_flag = False

        self.x_train, self.y_train = NumRecogTrainer._batch()
        self._start_keyboard_listener()

        self.losses = []
        self.mean_losses = []
        self.accuracies = []
        self.mean_accuracies = []

        self.threshold = 50

        self.loss_fn = get_lossfn(model.loss_function_name)[0]

    def _start_keyboard_listener(self):
        import keyboard
        def listen():
            keyboard.wait("q")
            self.stop_flag = True
        threading.Thread(target=listen, daemon=True).start()

    def _shuffle(x, y):
        perm = np.random.permutation(x.shape[1])
        return x[:,perm], y[:,perm]

    def _append_losses(self, y_pred, y):
        # ==== Compute and record loss
        l = self.loss_fn(y_pred, y)
        self.losses.append(l)

        # ==== Running average loss
        ml = np.mean(self.losses[max(0, len(self.losses) - self.threshold):])
        self.mean_losses.append(ml)

    def _append_accuracies(self, y_pred, y):
        # ==== Compute and record accuracy
        a = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y, axis=0))
        self.accuracies.append(a)

        # ==== Running average accuracies
        ma = np.mean(self.accuracies[max(0, len(self.accuracies) - self.threshold):])
        self.mean_accuracies.append(ma)

    def train(self, start=0, end=60000, batch=10, plot_interval=50):
        for i in range(start, end, batch):
            if self.stop_flag:
                raise TrainingHalt()
            
            epoch = (i - start) // batch
            
            # ==== Get training slices            
            x_slice = self.x_train[:, i:i+batch]
            y_slice = self.y_train[:, i:i+batch]

            self.model.backprop(x_slice, y_slice)
            y_pred = self.model.forward(x_slice)

            if epoch % plot_interval == 0:
                self._append_losses(y_pred, y_slice)
                self._append_accuracies(y_pred, y_slice)
                self._plot_metrics(epoch)

    def _plot_metrics(self, epoch):
        plt.clf()

        plt.suptitle(f"Epoch {epoch}")
        
        # Plot aclosscuracy
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(self.losses[max(0, len(self.losses) - self.threshold):], color="green", label="Current")
        plt.plot(self.mean_losses[max(0, len(self.mean_losses) - self.threshold):], color="blue", label="Mean")
        plt.xlabel("Recent Epochs")
        plt.ylabel(f"{self.model.loss_function_name} Loss")
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(self.accuracies[max(0, len(self.accuracies) - self.threshold):], color="green", label="Current")
        plt.plot(self.mean_accuracies[max(0, len(self.mean_accuracies) - self.threshold):], color="blue", label="Mean")
        plt.xlabel("Recent Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.pause(0.01)

    def save(self):
        self.model.save(f"{self.model.label}.npz")

    @staticmethod
    def _batch():
        from tensorflow.keras.datasets import mnist  # type: ignore
        print("Loading and preparing MNIST data...")

        TF_ENABLE_ONEDNN_OPTS=0

        (x_train, y_train), _ = mnist.load_data()
        x = x_train.reshape(-1, 784).T / 255.0  # Normalize and transpose to (784, N)
        y = np.eye(10)[y_train].T               # One-hot encode and transpose to (10, N)

        return NumRecogTrainer._shuffle(x, y)
    

    
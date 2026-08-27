from tensorflow.keras.datasets import mnist  # type: ignore
from main import NeuralNetwork
from main import Activator as af
import keyboard
import threading
import numpy as np
import matplotlib.pyplot as plt

# ==== Suppress ONEDNN optimization warning from TensorFlow
TF_ENABLE_ONEDNN_OPTS = 0

# ==== Initialize plot figures
#FIG_PRED = plt.figure("Prediction Viewer")  # For real-time digit display
FIG_METRICS = plt.figure("Metrics")            # For loss/accuracy metrics

# ==== Global flag for early stopping via keyboard
stop_training = False
def kb_listener():
    global stop_training
    keyboard.wait("q")
    stop_training = True
# Start the listener in a background thread
threading.Thread(target=kb_listener, daemon=True).start()

# ===== Show prediction result for a single image
def plot_prediciton(nn, x_sample, y_true):
    """
    x_sample: shape (784, 1) — single image column
    y_true: expected output (one-hot or index)
    """
    plt.clf()
    plt.figure(FIG_PRED.number)

    img = x_sample.reshape(28, 28)
    plt.imshow(img, cmap="gray")

    y_pred = nn.forward(x_sample)
    pred_label = np.argmax(y_pred)
    true_label = np.argmax(y_true)
    title = f"Prediction: {pred_label} | True: {true_label}"
    plt.title(title)
    plt.axis("off")
    plt.pause(0.05)  # Refresh the display

# ==== Plot training loss and accuracy over recent epochs
def plot_metrics(nn, 
             x_sample,
             y_true, 
             epoch,
             plot_iteration, 
             losses, 
             accuracies, 
             mean_accuracies, 
             mean_losses, 
             threshold=50):
    plt.clf()
    plt.figure(FIG_METRICS.number)
    plt.suptitle(f"Epoch {epoch}")

    # Compute and record loss
    y_pred = nn.forward(x_sample)
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)

    # Compute and record accuracy
    pred_labels = np.argmax(y_pred, axis=0)
    true_labels = np.argmax(y_true, axis=0)
    accuracy = np.mean(pred_labels == true_labels)
    accuracies.append(accuracy)

    # Running average noise
    mean_loss = np.mean(losses[max(0, plot_iteration - threshold):])
    mean_losses.append(mean_loss)


    # Running average accuracy 
    mean_accuracy = np.mean(accuracies[max(0, plot_iteration - threshold):])
    mean_accuracies.append(mean_accuracy)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(losses[max(0, plot_iteration - threshold):], color="red", label="Current")
    plt.plot(mean_losses[max(0, plot_iteration - threshold):], color="purple", label="Mean")
    plt.xlabel("Recent Epochs")
    plt.ylabel(f"{nn.loss_function.__name__} Loss")
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(accuracies[max(0, plot_iteration - threshold):], color="green", label="Current")
    plt.plot(mean_accuracies[max(0, plot_iteration - threshold):], color="blue", label="Mean")
    plt.xlabel("Recent Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.grid(True)

    plt.pause(0.05)
    return losses, accuracies

# ==== Training loop 
def number_recognition_trainer(nn,
                               x_train,
                               y_train,
                               start=0,
                               end=60000,
                               batch_size=10,
                               plot_step=50,
                            ):
    # Metrics history
    losses = [] 
    mean_losses = []
    accuracies = [] 
    mean_accuracies = []

    # Loop over training data in batches
    for i in range(start, end, batch_size):    
        epoch = (i - start) // batch_size

        # Exit early on 'q' press
        if stop_training:
            print("Process halted. Saving...")
            nn.save(f"{nn.label}.npz")
            break

        # Slice batch (keep column format: shape = (784, batch_size))
        x = x_train[:, i : i + batch_size]
        y = y_train[:, i : i + batch_size]

        # Backpropagation step
        nn.backprop(x, y)

        # Update metric plots
        if epoch % plot_step == 0:
            plot_iteration = epoch // plot_step
            losses, accuracy = plot_metrics(nn, x, y, epoch, plot_iteration, losses, accuracies, mean_accuracies, mean_losses)   
        #    plot_prediciton(nn, x_train[:, i:i+1], y_train[:, i:i+1])

    print("Training complete")
    # Save trained weights
    nn.save(f"{nn.label}.npz")

def shuffle(x_train, y_train):
    perm = np.random.permutation(x_train.shape[1])

    x_train_shuffled = x_train[:,perm]
    y_train_shuffled = y_train[:,perm]
    
    return x_train_shuffled, y_train_shuffled


# === Main Execution ===
if __name__ == "__main__":  
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Reshaping...") 
    # Flatten images to vectors and transpose to (784, N)
    x_train_reshaped = x_train.reshape(60000, 784).transpose()

    # One-hot encode labels and transpose to (10, N)
    y_train_onehot = np.eye(10)[y_train]
    y_train_reshaped = y_train_onehot.transpose()
    
    print("Shuffling...")
    x_train_shuffled, y_train_shuffled = shuffle(x_train_reshaped, y_train_reshaped)

    nn = NeuralNetwork.load("number_recognition_v1.1.npz")

    # Begin training from a certain index
    number_recognition_trainer(nn,
                               x_train_shuffled,
                               y_train_shuffled,
                               batch_size=8,
                               start=0)

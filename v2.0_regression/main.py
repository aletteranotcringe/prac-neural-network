from reg_trainer import RegTrainer, TrainingHalt
from modules.neuralnetwork import NeuralNetwork
from style import apply_dark_style
import numpy as np

def target_fn(x):
    return np.sin(x)

if __name__ == "__main__":
    apply_dark_style()        
    models = [NeuralNetwork.load("neural_network.npz")]             
    trainer = RegTrainer(models, target_fn, sample_range=(0, 2 * np.pi), batch_size=3)
    print("Press 'q' to stop training and save models.")

    try:
        trainer.train(epochs=500000, plot_interval=200)
    except TrainingHalt:     
        print("Training halted by user.")

    trainer.save_all()
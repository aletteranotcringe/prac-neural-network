from numrecog_trainer import NumRecogTrainer, TrainingHalt
from modules.neuralnetwork import NeuralNetwork
from style import apply_dark_style
import numpy as np

if __name__ == "__main__":
    apply_dark_style()        
    model = NeuralNetwork.load("numrecog.npz")
    model.learning_rate = 0.015

    trainer = NumRecogTrainer(model)
    print("Press 'q' to stop training and save models.")

    try:
        trainer.train(
            batch=5
        )
    except TrainingHalt:     
        print("Training halted by user.")

    trainer.save()
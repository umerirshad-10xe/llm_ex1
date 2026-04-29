# Author : Umer Irshad
# Date : 2026-04-21
# Module : LLM
# Section : Introduction
# Task Name : Coding Exercise Part 1
# Part : main
# Description : 

# This file is an entry point to intialize and run a given model architecture.

# Assumptions Made: The given Learning rate along with number of epochs is not 
# sufficient enough to obtain the desired 90%+ accuracy since the model is not
# complex enough to learn the features effectively for such a good accuracy. To 
# mitigate this problem, I have increased the learning rate and number of epochs
# to obtain 90%+ train and test accuracy.


import seaborn as sns
import numpy as np


from modules.layers import ConvolutionLayer, LinearLayer, Flatten
from modules.activation import ReLU, Softmax

from modules.loss import CrossEntropy
from modules.model import Model

from utils import (
    plot_confusion_matrix,
    plot_history,
    plot_predictions,
    load_dataset,
    compute_accuracy,
)

np.random.seed(0)
sns.set(style="darkgrid")
np.set_printoptions(precision=4, suppress=True)

EPOCHS = 50
LEARNING_RATE = 0.1
BATCH_SIZE = 32


layer_list = [
    ConvolutionLayer(in_channels=1, out_channels=3, kernel_size=5),
    ReLU(),
    Flatten(),
    LinearLayer(48, 10),
    Softmax(),
]


def main():

    x_train, x_test, y_train, y_test = load_dataset()

    model = Model(layer_list, CrossEntropy())

    print("-" * 80)
    print(
        f"\tTrainig Model: Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}"
    )
    print("-" * 80)

    history = model.train(
        x_train,
        y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    print("... Traning Complete")

    predictions = model.predict(x_test)
    accuracy = compute_accuracy(predictions, y_test)
    print("==============================")
    print(f"\tAccuracy: {accuracy:.2%}")
    print("==============================")

    plot_history(history)
    plot_predictions(x_test, predictions, accuracy)
    plot_confusion_matrix(y_test, predictions)


if __name__ == "__main__":
    main()

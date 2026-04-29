# Author : Umer Irshad
# Date : 2026-04-21
# Module : LLM
# Section : Introduction
# Task Name : Coding Exercise Part 1
# Part : Activations
# Description : 

# This file contains the 3 activations being used in the NNs:
#   1. ReLU (Rectified Linear Unit)
#   2. Sigmoid
#   3. Softmax
# Forward and backward pass for each layer is implemented.
# For softmax, the derivative is calcualted by reducing 
# the expressions obtained from the jacobian matrix multiplication


import numpy as np


class ReLU:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        return np.maximum(0, inputs)
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the ReLU activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """
        # ================ Insert Code Here ================
        return {"d_out": d_outputs * (self.inputs > 0)}
        # ==================================================


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the Sigmoid activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        return 1 / (1 + np.exp(-inputs))
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Sigmoid activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        sig = 1 / (1 + np.exp(-self.inputs))
        return {"d_out": d_outputs * sig * (1 - sig)}
        # ==================================================


class Softmax:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """
        self.inputs = inputs

        # ================ Insert Code Here ================
        # Shifting by max for numerical stability
        exp_shifted = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Softmax activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        softmax = self.forward(self.inputs)
        dot = np.sum(d_outputs * softmax, axis=-1, keepdims=True)
        d_inputs = softmax * (d_outputs - dot)

        return {"d_out": d_inputs}
        # ==================================================

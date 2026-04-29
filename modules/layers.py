# Author : Umer Irshad
# Date : 2026-04-21
# Module : LLM
# Section : Introduction
# Task Name : Coding Exercise Part 1
# Part : Layers
# Description : 

# This file contains the following Layers of a deep neural network: 
#   1. Convolution Layer
#   2. Liner Layer

# Convolution Layer takes in the input and output channels. The provided
# kernel size is a single number which is translated into a square shaped
# filter to be applied. The stride parameter decides the stride decided 
# vertically and horizontally to perform the convolution operation.

# Liner Layer is by definition a fully connected layer with parameters of 
# input features and output features. The weight matrix of such a layer contains
# output_features * input_features weights.

# The weights for each layer is being initialized using kaiming intilization. 

import numpy as np


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)

        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * std

        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a convolution layer

        Args:
            inputs (np.ndarray):
                array of shape
                (batch_size, in_channels, height, width)

        Returns: (np.ndarray):
            array of shape
            (batch_size, out_channels, new_height, new_width)
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        batch_size, in_channels, H, W = inputs.shape
        k = self.kernel_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        outputs = np.zeros((batch_size, self.out_channels, out_H, out_W), dtype=np.float32)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * s
                        w_start = j * s
                        region = inputs[b, :, h_start:h_start+k, w_start:w_start+k]
                        outputs[b, oc, i, j] = np.sum(region * self.weights[oc]) + self.bias[oc]

        return outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of convolution layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_channels, new_height, new_width)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"

        """

        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        # ================ Insert Code Here ================
        inputs = self.inputs
        batch_size, in_channels, H, W = inputs.shape
        k = self.kernel_size
        s = self.stride

        _, _, out_H, out_W = d_outputs.shape

        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_inputs = np.zeros_like(inputs)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                d_bias[oc] += np.sum(d_outputs[b, oc])

                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * s
                        w_start = j * s

                        region = inputs[b, :, h_start:h_start+k, w_start:w_start+k]

                        d_weights[oc] += d_outputs[b, oc, i, j] * region
                        d_inputs[b, :, h_start:h_start+k, w_start:w_start+k] += (
                            d_outputs[b, oc, i, j] * self.weights[oc]
                        )

        return {
            "d_weights": d_weights,
            "d_bias": d_bias,
            "d_out": d_inputs,
        }
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================


class Flatten:
    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, d_outputs):
        return {"d_out": d_outputs.reshape(self.inputs_shape)}


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        fan_in = in_features
        std = np.sqrt(2.0 / fan_in)

        self.weights = np.random.randn(out_features, in_features).astype(np.float32) * std
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a linear layer

        Args:
            inputs (np.ndarray):
                array of shape (batch_size, in_features)

        Returns: (np.ndarray):
            array of shape (batch_size, out_features)
        """

        # ================ Insert Code Here ================
        self.inputs = inputs
        return inputs @ self.weights.T + self.bias
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of Linear layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_features)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"
        """
        if self.inputs is None:
            raise NotImplementedError("Need to call forward function before backward function")
        
        # ================ Insert Code Here ================
        inputs = self.inputs

        d_weights = d_outputs.T @ inputs
        d_bias = np.sum(d_outputs, axis=0)
        d_inputs = d_outputs @ self.weights

        return {
            "d_weights": d_weights,
            "d_bias": d_bias,
            "d_out": d_inputs,
        }
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================

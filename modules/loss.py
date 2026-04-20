import numpy as np


class CrossEntropy:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.inputs = None
        self.targets = None

    def forward(self, inputs, targets):
        """Forward pass for the cross entropy loss function

        Args:
            inputs (np.ndarray): predictions from the model
            targets (_type_): ground truth labels (one-hot encoded)

        Returns: (float):
            loss value
        """

        self.inputs = inputs
        self.targets = targets

        # ================ Insert Code Here ================
        inputs_clipped = np.clip(inputs, self.eps, 1.0 - self.eps)
        loss = -np.sum(targets * np.log(inputs_clipped)) / inputs.shape[0]
        return loss
        # ==================================================

    def backward(self):
        """Backward pass for the cross entropy loss function

        Args:
            None

        Returns: (dict):
            Dictionary containing the derivative of the loss
            with respect to the inputs to the loss function.
            The key of the dictionary should be "d_out"
        """
        
        # ================ Insert Code Here ================
        inputs_clipped = np.clip(self.inputs, self.eps, 1.0 - self.eps)
        d_out = (self.targets / inputs_clipped) / self.inputs.shape[0] * -1

        return {"d_out": d_out}

        # ==================================================

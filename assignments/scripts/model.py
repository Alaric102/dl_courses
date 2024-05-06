import numpy as np
from typing import Dict

from scripts.layers import FullyConnectedLayer, ReLULayer, Param, softmax_with_cross_entropy, l2_regularization

class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc_1 = FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size)
        self.relu = ReLULayer()
        self.fc_2 = FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        # Before running forward/backward pass clear gradients
        self.params()["W1"].grad = np.zeros_like(self.params()["W1"].value)
        self.params()["B1"].grad = np.zeros_like(self.params()["B1"].value)
        self.params()["W2"].grad = np.zeros_like(self.params()["W2"].value)
        self.params()["B2"].grad = np.zeros_like(self.params()["B2"].value)
        
        # Forward pass
        loss, d_pred = softmax_with_cross_entropy(
            self.fc_2.forward(
                self.relu.forward(
                    self.fc_1.forward(X))), y)
        
        # Backward pass
        self.fc_1.backward(
            self.relu.backward(
                self.fc_2.backward(d_pred)))

        # L2 regularization on all params
        fc1_W_loss, fc1_W_dreg = l2_regularization(self.fc_1.params()["W"].value, self.reg)
        fc1_B_loss, fc1_B_dreg = l2_regularization(self.fc_1.params()["B"].value, self.reg)
        fc2_W_loss, fc2_W_dreg = l2_regularization(self.fc_2.params()["W"].value, self.reg)
        fc2_B_loss, fc2_B_dreg = l2_regularization(self.fc_2.params()["B"].value, self.reg)

        loss = loss + fc1_W_loss + fc1_B_loss + fc2_W_loss + fc2_B_loss
        self.params()["W1"].grad += fc1_W_dreg
        self.params()["B1"].grad += fc1_B_dreg
        self.params()["W2"].grad += fc2_W_dreg
        self.params()["B2"].grad += fc2_B_dreg
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        result = np.zeros(X.shape[0], int)
        predictions = self.fc_2.forward(self.relu.forward(self.fc_1.forward(X)))
        result = np.argmax(predictions, axis=1)
        return result

    def params(self) -> Dict[str, Param]:
        result = {"W1": self.fc_1.params()["W"],
                  "B1": self.fc_1.params()["B"],
                  "W2": self.fc_2.params()["W"],
                  "B2": self.fc_2.params()["B"]}
        return result

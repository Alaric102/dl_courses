import numpy as np
from typing import Dict

from scripts.layers import ConvolutionalLayer, MaxPoolingLayer, Flattener, FullyConnectedLayer, ReLULayer, Param, softmax_with_cross_entropy, l2_regularization
from scripts.linear_classifer import softmax
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

class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        
        image_width, image_height, n_channels = input_shape
        conv1_out_channels = 64
        self.conv2d_1 = ConvolutionalLayer(filter_size=conv1_channels, in_channels=n_channels, out_channels=conv1_out_channels)
        self.relu_1 = ReLULayer()
        self.maxpool_1 = MaxPoolingLayer(pool_size=3, stride=2)

        conv2_out_channels = 16
        self.conv2d_2 = ConvolutionalLayer(filter_size=conv2_channels, in_channels=conv1_out_channels, out_channels=conv2_out_channels)
        self.relu_2 = ReLULayer()
        self.maxpool_2 = MaxPoolingLayer(pool_size=3, stride=1)

        self.flattener = Flattener()
        self.fc_layer = FullyConnectedLayer(n_input=12*12*16, n_output=n_output_classes)
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        t1 = self.maxpool_1.forward(self.relu_1.forward(self.conv2d_1.forward(X)))
        t2 = self.maxpool_2.forward(self.relu_2.forward(self.conv2d_2.forward(t1)))
        t3 = self.flattener.forward(t2)
        loss, grad = softmax_with_cross_entropy(self.fc_layer.forward(t3), target_index=y)

        dt2 = self.flattener.backward(self.fc_layer.backward(grad))
        dt1 = self.conv2d_2.backward(self.relu_2.backward(self.maxpool_2.backward(dt2)))
        self.conv2d_1.backward(self.relu_1.backward(self.maxpool_1.backward(dt1)))

        return loss
    
    def predict(self, X):
        # You can probably copy the code from previous assignment
        preds = self.maxpool_1.forward(self.relu_1.forward(self.conv2d_1.forward(X)))
        preds = self.maxpool_2.forward(self.relu_2.forward(self.conv2d_2.forward(preds)))
        preds = self.fc_layer.forward(self.flattener.forward(preds))

        probs = softmax(preds)
        return np.argmax(probs, axis=1)

    def params(self):
        result = {
            'conv2d_1.W': self.conv2d_1.W, 'conv2d_1.B': self.conv2d_1.B, 
            'conv2d_2.W': self.conv2d_2.W, 'conv2d_2.B': self.conv2d_2.B,
            'fc.W': self.fc_layer.W, 'fc.B': self.fc_layer.B
        }
        return result

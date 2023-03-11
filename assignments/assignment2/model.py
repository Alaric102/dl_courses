import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


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
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")
        params = self.params()
        params["W1"].grad = np.zeros_like(params["W1"].value)
        params["B1"].grad = np.zeros_like(params["B1"].value)
        params["W2"].grad = np.zeros_like(params["W2"].value)
        params["B2"].grad = np.zeros_like(params["B2"].value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        l1_res = self.layer1.forward(X)
        relu_res = self.relu.forward(l1_res)
        l2_res = self.layer2.forward(relu_res)
        loss, dpred = softmax_with_cross_entropy(l2_res, y)

        l2_grad = self.layer2.backward(dpred)
        relu_grad = self.relu.backward(l2_grad)
        self.layer1.backward(relu_grad)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        layer1_w_reg_loss, layer1_w_dreg = l2_regularization(self.layer1.params()["W"].value, self.reg)
        layer1_b_reg_loss, layer1_b_dreg = l2_regularization(self.layer1.params()["B"].value, self.reg)
        layer2_w_reg_loss, layer2_w_dreg = l2_regularization(self.layer2.params()["W"].value, self.reg)
        layer2_b_reg_loss, layer2_b_dreg = l2_regularization(self.layer2.params()["B"].value, self.reg)
        
        loss += layer1_w_reg_loss + layer1_b_reg_loss + layer2_w_reg_loss + layer2_b_reg_loss
        params["W1"].grad += layer1_w_dreg
        params["B1"].grad += layer1_b_dreg
        params["W2"].grad += layer2_w_dreg
        params["B2"].grad += layer2_b_dreg
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
        pred = np.zeros(X.shape[0], np.int)

        preds = self.layer2.forward(self.relu.forward(self.layer1.forward(X)))

        pred = np.argmax(preds, axis=1)
        # raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {"W1": self.layer1.params()["W"],
                  "B1": self.layer1.params()["B"],
                  "W2": self.layer2.params()["W"],
                  "B2": self.layer2.params()["B"]}

        # TODO Implement aggregating all of the params

        return result

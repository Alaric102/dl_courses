import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    a = np.sum(W**2)
    f = reg_strength * a
    dfda = reg_strength
    dfdw = dfda * 2*W
    return f, dfdw


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    gt_indexes = target_index
    preds = predictions
    if type(gt_indexes) == int:
      gt_indexes = [gt_indexes]
      preds = preds[None, :]
    
    batch_size = preds.shape[0]
    preds = preds - np.max(preds, axis=1, keepdims=True)
    a = np.exp(preds)
    b = np.sum(a, axis=1, keepdims=True)
    c = a / b
    d = np.log(c)
    t = np.zeros_like(a)
    for row, col in enumerate(gt_indexes):
      t[row, col] = 1
    e = t*d
    f = np.sum(-e)/batch_size

    dfde = -batch_size/(batch_size**2)
    dfdd = dfde * t
    dfdc = dfdd * 1/c
    dfdb = np.sum(dfdc * (-a / (b ** 2)), axis=1, keepdims=True)
    dfda = dfdc * (1/b) + dfdb
    dfdx = dfda * a

    dfdx = np.reshape(dfdx, predictions.shape)
    return f, dfdx


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = np.copy(X)
        result = np.copy(X)
        result[X<0] = 0
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.copy(d_out)
        d_result[self.X < 0] *= 0
        d_result[self.X == 0] *= 0.5
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        return (X @ self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # out = X @ W + B
        self.X.grad = d_out @ self.W.value.T
        
        # and gradients with respect to W and B
        dW = self.X.value.T @ d_out
        dB = np.sum(d_out, axis=0, keepdims=True)
        # Add gradients of W and B to their `grad` attribute
        self.W.grad = dW
        self.B.grad = dB
        self.X.grad = d_out @ self.W.value.T
        # It should be pretty similar to linear classifier from
        
        return self.X.grad

    def params(self):
        return {'W': self.W, 'B': self.B}

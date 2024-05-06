import numpy as np


def l2_regularization(W: np.ndarray, reg_strength: float):
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
    loss = reg_strength * a
    dfda = reg_strength
    grad = dfda * 2*W
    return loss, grad


def softmax_with_cross_entropy(predictions: np.ndarray, target_index: np.ndarray):
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
    preds = predictions.copy()
    gt_indices = target_index
    preds -= np.max(preds, axis=1, keepdims=True)
    batch_size, _ = preds.shape

    a = np.exp(preds)
    b = np.sum(a, axis=1, keepdims=True)
    c = a/b
    d = np.log(c)
    t = np.zeros_like(d)
    for row, col in zip(range(batch_size), gt_indices):
      t[row, col] = 1
    e = t*d
    f = np.sum(e, axis=1)
    g = np.sum(-f)/batch_size

    dgdf = -batch_size/batch_size**2
    dgde = dgdf * 1
    dgdd = dgde * t
    dgdc = dgdd * 1/c
    dgdb = dgdc * (-a/b**2)
    dgdc_ext = dgdc
    dgdc_ext[t == 0] = 1/batch_size
    dgda = dgdb + dgdc_ext * 1/b
    dgdx = dgda * a

    loss = g
    d_preds = dgdx
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer():
    def __init__(self):
        pass

    def forward(self, X: np.ndarray):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X.copy()
        result = X.copy()
        result[result < 0] = 0
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
        # Your final implementation shouldn't have any loops
        d_result = np.ones_like(self.X)
        d_result[self.X < 0] = 0
        return d_result * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        return np.dot(X, self.W.value) + self.B.value

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
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        # https://cs231n.stanford.edu/handouts/linear-backprop.pdf
        self.X.grad = d_out @ self.W.value.T
        self.W.grad = self.X.value.T @ d_out
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)

        # It should be pretty similar to linear classifier from
        # the previous assignment
        return self.X.grad

    def params(self):
        return {'W': self.W, 'B': self.B}

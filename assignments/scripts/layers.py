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

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding: int = 0):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X: np.ndarray):
        batch_size, height, width, channels = X.shape
        X_with_padding = np.zeros((batch_size, height+2*self.padding, width+2*self.padding, channels))
        X_with_padding[:, self.padding:self.padding+height, self.padding:self.padding+width,:] = X
        # batch_size, height, width, channels = X_with_padding.shape
        out_height = height - self.filter_size + 1 + 2*self.padding
        out_width = width - self.filter_size + 1 + 2*self.padding
        
        self.X = Param(X_with_padding)
        out = np.zeros((batch_size, out_height, out_width, self.out_channels), dtype=float)
        for y in range(out_height):
            for x in range(out_width):
                X_roi = X_with_padding[:, y:y+self.filter_size, x:x+self.filter_size, :]                                # [bs,  h,  w, ch]
                X_flatten = X_roi.reshape(batch_size, self.filter_size*self.filter_size*channels)                       # [bs, fs, fs, ch]
                W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
                out[:, y, x, :] = X_flatten @ W_flatten + self.B.value
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape

        for y in range(out_height):
            for x in range(out_width):
                dvalue = d_out[:, y, x, :]
                W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
                dX_flatten = dvalue @ W_flatten.T
                self.X.grad[:, y:y+self.filter_size, x:x+self.filter_size, :] += dX_flatten.reshape(batch_size, self.filter_size, self.filter_size, channels)

                X_flatten = self.X.value[:, y:y+self.filter_size, x:x+self.filter_size, :].reshape(batch_size, self.filter_size*self.filter_size*channels)
                dW_flatten = X_flatten.T @ dvalue
                self.W.grad += dW_flatten.reshape(self.W.value.shape)
                
                self.B.grad += np.sum(dvalue, axis=0)
        return self.X.grad[:, self.padding:height-self.padding, self.padding:width-self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, channels))
        self.X = X.copy()
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                out[:, y, x, :] = np.amax(X_slice, axis=(1, 2))
        return out
    
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask
        return out
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)
    
    def params(self):
        # No params!
        return {}

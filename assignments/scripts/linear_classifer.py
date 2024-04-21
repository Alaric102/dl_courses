import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # Your final implementation shouldn't have any loops
    if len(predictions.shape) == 2:
      preds = predictions - np.max(predictions, axis=1, keepdims=True)
      return np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    preds = predictions - np.max(predictions)
    return np.exp(preds) / np.sum(np.exp(preds))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    # Your final implementation shouldn't have any loops
    true_prob = np.zeros_like(probs)
    true_prob[target_index] = 1
    return -np.sum(true_prob * np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # Your final implementation shouldn't have any loops
    # Prepare no batch input data assuming that it's batch_size = 1
    preds = predictions.copy()
    gt_indices = target_index
    if type(gt_indices) is int:
      preds = preds[None, :]
      gt_indices = np.array([[gt_indices]])
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
    dprediction = dgdx
    if type(target_index) is int:
        dprediction = dprediction[0]
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    a = np.sum(W**2)
    loss = reg_strength * a
    dfda = reg_strength
    grad = dfda * 2*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dlossdx = softmax_with_cross_entropy(predictions, target_index)
    # Your final implementation shouldn't have any loops
    # TODO: figure out formula below
    dW = X.T @ dlossdx
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self, is_verbose: bool = False):
        self.W = None
        self.is_verbose_ = is_verbose

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        # if self.W is None:
        self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for indices in batches_indices:
              X_batch = X[indices]
              y_batch = y[indices]
              loss_l1, dW_l1 = linear_softmax(X_batch, self.W, y_batch)
              loss_l2, dW_l2 = l2_regularization(self.W, reg)
              loss = loss_l1 + loss_l2
              dW = dW_l1 + dW_l2
              self.W -= learning_rate * dW
              loss_history.append(loss)
            # end
            loss = np.mean(loss_history)
            if self.is_verbose_:
              print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred



                
                                                          

            

                

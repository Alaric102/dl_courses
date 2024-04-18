import numpy as np
import scipy.io as io
import os

def load_data_mat(filename: str, max_samples: int, seed : int = 42):
    '''
    Loads numpy arrays from .mat file

    Returns:
    X, np array (num_samples, 32, 32, 3) - images
    y, np array of int (num_samples) - labels
    '''
    raw = io.loadmat(filename)
    X = raw['X']  # Array of [32, 32, 3, n_samples]
    y = raw['y']  # Array of [n_samples, 1]
    X = np.moveaxis(X, [3], [0])
    y = y.flatten()
    # Fix up class 0 to be 0
    y[y == 10] = 0

    np.random.seed(seed)
    samples = np.random.choice(np.arange(X.shape[0]),
                               max_samples,
                               replace=False)
    
    return X[samples].astype(np.float32), y[samples]

def load_svhn(folder: str, max_train: int, max_test: int):
    '''
    Loads SVHN dataset from file

    Arguments:

    Returns:
    train_X, np array (num_train, 32, 32, 3) - training images
    train_y, np array of int (num_train) - training labels
    test_X, np array (num_test, 32, 32, 3) - test images
    test_y, np array of int (num_test) - test labels
    '''
    train_X, train_y = load_data_mat(os.path.join(folder, "train_32x32.mat"), max_train)
    test_X, test_y = load_data_mat(os.path.join(folder, "test_32x32.mat"), max_test)
    return train_X, train_y, test_X, test_y

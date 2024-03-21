import numpy as np
from scipy.optimize import minimize
from data_processing.postprocessing import monomial_power
import math


def standardize_psi(psi,h, derivative):
    if derivative not in ['dtdx', 'dtdy', 'Laplace']:
        raise ValueError("Invalid derivative type")

    if derivative == 'Laplace':
        h = h**2
    psi = psi * h
    return psi


def create_train_test(features, labels, tt_split=0.9, seed=None):
    if seed is not None:
        np.random.seed(seed)

    rows = features.shape[0]
    train_size = int(rows * tt_split)

    train_index = np.random.choice(rows, train_size, replace=False)

    test_index = np.setdiff1d(np.arange(rows), train_index)

    train_f = features[train_index]
    train_f = train_f.reshape(train_f.shape[0], -1)

    test_f = features[test_index]
    test_f = test_f.reshape(test_f.shape[0], -1)

    train_l = labels[train_index]
    test_l = labels[test_index]

    return train_f, train_l, test_f, test_l, train_index, test_index



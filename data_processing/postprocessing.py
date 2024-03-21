import numpy as np
import math


def rescale_psi(pred_psi, actual_psi, h, derivative):
    if derivative not in ['dtdx', 'dtdy', 'Laplace']:
        raise ValueError("Invalid derivative type")

    if derivative == 'Laplace':
        h = h**2

    scaled_psi_act = actual_psi / h
    scaled_psi_pred = pred_psi / h

    return scaled_psi_pred, scaled_psi_act


def error_test_func(scaled_feat, scaled_w):
    error = []
    for i in range(scaled_feat.shape[0]):
        temp = 0
        for j in range(scaled_feat.shape[1]):
            temp = ((scaled_feat[i, j, 0] ** 2 / 2 + scaled_feat[i, j, 1] ** 2 / 2) * scaled_w[i, j]) + temp
        error.append(temp)
    return np.array(error)


def monomial_power(polynomial):
    """

    :param polynomial:
    :return:
    """
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_moments(test_f, test_l, polynomial):
    n = int((polynomial ** 2 + 3 * polynomial) / 2)
    test_f = test_f.reshape((test_f.shape[0], n, n))
    test_l = test_l[:, :, np.newaxis]
    moments = np.matmul(test_f, test_l)
    return moments.squeeze(-1)


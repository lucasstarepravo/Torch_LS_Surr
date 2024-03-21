import numpy as np
import math


# This will be used to rescale the features and labels (and the predicted label)
def rescale_stencil(actual_l, pred_l, stand_feat, f_mean, f_stdv, l_mean, l_stdv, test_index):
    '''
    This function is supposed to take a subset of the orignal dataset (only the train or test set), and the corresponding
    features dataset (thus, if inputting the test predicted and actual weights, the test features data must be input).
    The data is normalized before inputting to the ANN. In this function the data is rescaled back to
    its original magnitude and distribution.
    :param actual_l: the dataset containing the actual normalized weights
    :param pred_l: the datset containing the predicted normalized weights (ANN output)
    :param stand_feat: Is the normalized features corresponding dataset
    :param f_mean: Is the mean of the features (the complete dataset, NOT only the test means)
    :param f_stdv: Is the standard deviation of the features (the complete dataset, NOT only the test standard deviation)
    :param l_mean: Is the mean of the labels (the complete dataset, NOT only the test means)
    :param l_stdv: Is the standard deviation of the labels (the complete dataset, NOT only the test standard deviation)
    :param test_index: Is the index of rows of features and labels that were separated for the test dataset
    :return:
    '''
    tst_mean_l = l_mean[test_index]
    tst_std_l = l_stdv[test_index]

    scaled_actual_l = actual_l * tst_std_l + tst_mean_l
    scaled_pred_l = pred_l * tst_std_l + tst_mean_l

    tst_mean_f = f_mean[test_index]
    tst_std_f = f_stdv[test_index]

    scaled_feat = stand_feat.reshape(stand_feat.shape[0], -1, 2) * tst_std_f + tst_mean_f

    return scaled_actual_l, scaled_pred_l, scaled_feat


def d_2_c(coor, test_index, scaled_feat):
    '''
    The ANN features are the x and y distances of the neighbour nodes to the reference node. This function takes the
    whole coordinates original vector, and the test_index vector obtained from the tran_test_split and, finds the
    coordinates of the test nodes
    :param coor:
    :param test_index:
    :param scaled_feat:
    :return:
    '''
    zeros = np.zeros((int(scaled_feat.shape[0] * 2))).reshape(scaled_feat.shape[0], -1, 2)
    scaled_feat = np.concatenate((zeros, scaled_feat), axis=1)
    tst_coor = coor[test_index, :]
    tst_coor = tst_coor.reshape(tst_coor.shape[0], -1, 2)
    d_2_c = scaled_feat + tst_coor
    return d_2_c


def rescale_h(actual_l, pred_l, feat_subset, f_mean, l_mean, h_scale_xy, h_scale_w, test_index):

    sc_actual_l = actual_l / h_scale_w + l_mean
    sc_pred_l = pred_l / h_scale_w + l_mean

    tst_mean_f = f_mean[test_index]

    sc_feat = feat_subset.reshape(feat_subset.shape[0], -1, 2) * h_scale_xy + tst_mean_f

    return sc_actual_l, sc_pred_l, sc_feat


def rescale_global_stand(actual_l, pred_l, feat_subset, f_mean, f_stdv, l_mean, l_stdv, test_index):
    scaled_actual_l = actual_l * l_stdv + l_mean
    scaled_pred_l = pred_l * l_stdv + l_mean

    tst_mean_f = f_mean[test_index]
    tst_std_f = f_stdv[test_index]

    scaled_feat = feat_subset.reshape(feat_subset.shape[0], -1, 2) * tst_std_f + tst_mean_f

    return scaled_actual_l, scaled_pred_l, scaled_feat


def rescale_std(actual_l, pred_l, feat_subset, f_mean, f_stdv, l_std, test_index):

    tst_mean_f = f_mean[test_index]
    tst_std_f = f_stdv[test_index]
    tst_std_l = l_std[test_index]

    scaled_actual_l = actual_l / tst_std_l
    scaled_pred_l = pred_l / tst_std_l

    scaled_feat = feat_subset.reshape(feat_subset.shape[0], -1, 2) * tst_std_f + tst_mean_f

    return scaled_actual_l, scaled_pred_l, scaled_feat


def rescale_spread(actual_l, pred_l, feat_subset, f_stdv, l_std, test_index, polynomial):
    tst_std_f = f_stdv[test_index]
    tst_std_l = l_std[test_index]

    scaled_actual_l = actual_l / tst_std_l
    scaled_pred_l = pred_l / tst_std_l

    n = int((polynomial ** 2 + 3 * polynomial) / 2)
    feat_subset = feat_subset.reshape(feat_subset.shape[0], -1, n)
    scaled_feat = feat_subset[:, :, :2] * tst_std_f

    return scaled_actual_l, scaled_pred_l, scaled_feat


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


def calc_moments(neigh_xy_d, scaled_w, polynomial):
    mon_power = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in mon_power:
        monomial.append((neigh_xy_d[:, :, 0] ** power_x * neigh_xy_d[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    moments = np.array(monomial) * scaled_w
    moments = np.sum(moments, axis=2)
    return moments.T

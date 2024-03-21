import numpy as np
from scipy.optimize import minimize
from learning_shapefunction.data_processing.postprocessing import monomial_power
import math


def feat_extract(coor, neigh_link):
    """

    :param coor:
    :param neigh_link:
    :return:
    features: is a np.array with 3D dimensions (ref_node_index, neigh_node_index, x_or_y_distance from ref node)
    """
    neigh_link = neigh_link - 1
    rows = neigh_link.shape[0]
    cols = neigh_link.shape[1]
    features = []
    for i in range(rows):
        temp_list_f = []
        for j in range(cols):
            x_dist = coor[int(neigh_link[i, j]), 0] - coor[int(neigh_link[i, 0]), 0]
            y_dist = coor[int(neigh_link[i, j]), 1] - coor[int(neigh_link[i, 0]), 1]
            temp_list_f.append(tuple([x_dist, y_dist]))
        features.append(temp_list_f)
    return np.array(features)


def non_dimension(features, labels, dx, dtype='laplace'):
    '''
    This function uses the stencil size which is 1.5dx to normalize the feature vector
    :param features:
    :param labels:
    :param dx:
    :param dtype:
    :return:
    '''
    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')
    dx = 1.5 * dx  # The 1.5 is obtained from the Fortran code, which is the ratio between h/s
    if dtype == 'laplace':
        h_scale_w = dx ** 2
        h_scale_xy = dx
    else:
        h_scale_w = dx
        h_scale_xy = dx

    f_mean = np.mean(features, axis=1, keepdims=True)
    stand_feature = (features - f_mean) / h_scale_xy

    l_mean = np.mean(labels)
    stand_label = (labels - l_mean) * h_scale_w
    return stand_feature, stand_label, f_mean, l_mean, h_scale_xy, h_scale_w


def standardize_comp_stencil(features, labels):
    f_sten_mean = np.mean(features, axis=1, keepdims=True)
    f_sten_std = np.std(features, axis=1, keepdims=True)
    stand_feature = (features - f_sten_mean) / f_sten_std

    l_sten_mean = np.mean(labels, axis=1, keepdims=True)
    l_sten_std = np.std(labels, axis=1, keepdims=True)
    stand_label = (labels - l_sten_mean) / l_sten_std

    return stand_feature, stand_label, f_sten_mean, f_sten_std, l_sten_mean, l_sten_std


def global_standard(features, labels):
    f_sten_mean = np.mean(features, axis=1, keepdims=True)
    f_sten_std = np.std(features, axis=1, keepdims=True)
    stand_features = (features - f_sten_mean) / f_sten_std

    l_mean = np.mean(labels)
    l_stdv = np.std(labels)
    stand_labels = (labels - l_mean) / l_stdv

    return stand_features, stand_labels, f_sten_mean, f_sten_std, l_mean, l_stdv


def std_dev_norm(features, labels, dtype):

    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')

    f_sten_mean = np.mean(features, axis=1, keepdims=True)
    f_sten_std = np.std(features, axis=1, keepdims=True)
    stand_features = (features - f_sten_mean) / f_sten_std

    if dtype == 'laplace':
        l_std = np.mean(f_sten_std, axis=2)
        l_std = l_std ** 2
    else:
        l_std = np.mean(f_sten_std, axis=2)

    stand_labels = labels * l_std

    return stand_features, stand_labels, f_sten_mean, f_sten_std, l_std


def spread_norm(features, weights, dtype):
    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')
    f_sten_std = np.std(features, axis=1, keepdims=True)
    stand_features = features/f_sten_std

    if dtype == 'laplace':
        l_std = np.mean(f_sten_std, axis=2)
        l_std = l_std ** 2
    else:
        l_std = np.mean(f_sten_std, axis=2)

    stand_labels = weights * l_std
    return stand_features, stand_labels, f_sten_std, l_std


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


def trim_zero_columns(array, tolerance=1e-10):
    # Iterate through each column and check if all elements are effectively zero
    for col_index in range(array.shape[1]):
        if np.all(np.isclose(array[:, col_index], 0, atol=tolerance)):
            # Return the array sliced up to the current column
            return array[:, :col_index]
    return array  # Return the original array if no all-zero column is found


def monomial_expansion(features, polynomial):
    monomial_exponent = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in monomial_exponent:
        monomial.append((features[:, :, 0] ** power_x * features[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    return np.transpose(np.array(monomial), (1, 2, 0))


'''Functions below are used to get information about the average weights given the average node distance'''

def avg_distance(features):
    avg_dist = (np.mean(abs(features), axis=1, keepdims=True))
    avg_dist = np.mean(avg_dist, axis=2)
    return avg_dist


def avg_weight(weights):
    avg_weigh = np.mean(weights, axis=1, keepdims=True)
    return avg_weigh


def coefficient_opt_c(x_values, y_values):
    '''This function will be used to optimise the c coefficient to link the '''
    # Function to calculate squared errors for given coefficient 'c'
    def squared_error(c, x_values, y_values):
        y_predicted = (1 / (c * x_values)) ** 2
        return np.sum((y_values - y_predicted) ** 2)

    # Objective function to minimize (wrapper for the squared_error function)
    def objective_function(c):
        return squared_error(c, x_values, y_values)

    # Initial guess for 'c'
    initial_guess = [2.0]

    # Minimize the squared error to find the optimal 'c'
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')

    # Output the result
    optimal_c = result.x[0]
    return optimal_c


def coefficient_opt_capla(x_values, y_values, features):
    '''This function will optimize both c and alpha coefficients.'''
    '''
    x_values are the average s_distances for each node
    y_values are the average weights for each node'''
    # Function to calculate squared errors for given coefficients 'c' and 'alpha'

    var_x = np.var(features[:, :, 0], axis=1)
    var_y = np.var(features[:, :, 1], axis=1)

    def squared_error(params, x_values, y_values, var_x, var_y):
        c, alpha = params
        y_predicted = (1 / (c * x_values)) ** 2 + alpha * (var_x + var_y)
        return np.sum((y_values - y_predicted) ** 2)

    # Objective function to minimize
    def objective_function(params):
        return squared_error(params, x_values, y_values, var_x, var_y)

    # Initial guesses for 'c' and 'alpha'
    initial_guess = [2.0, 0.1]

    # Minimize the squared error to find the optimal 'c' and 'alpha'
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')

    # Output the result
    optimal_c, optimal_alpha = result.x
    return optimal_c, optimal_alpha



def evaluate_model_error(x_values, y_actual, optimal_c):
    '''This should be used to evaluate the model attempting to capture the raltionship betweent the average distance
    and the average weight'''
    y_predicted = (1 / (optimal_c * x_values))**2
    errors = ((y_actual - y_predicted) ** 2)**.5
    total_error = np.mean(errors)
    return errors, total_error


def evaluate_model_error_alpha(x_values, y_actual, features, optimal_c, optimal_alpha):
    '''This should be used to evaluate the model attempting to capture the raltionship between the average distance
    and the average weight'''
    var_x = np.var(features[:, :, 0], axis=1)
    var_y = np.var(features[:, :, 1], axis=1)
    y_predicted = (1 / (optimal_c * x_values))**2
    errors = ((y_actual - y_predicted) ** 2)**.5 + optimal_alpha * (var_x + var_y)
    total_error = np.mean(errors)
    return errors, total_error

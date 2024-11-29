import numpy as np
import logging
import os
from models.SaveNLoad import *
import pickle as pk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def save_variable_with_pickle(variable, variable_name, variable_id, filepath):
    # Ensure the directory exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Construct the filename with the ID appended
    file_name = f"{variable_name}{variable_id}.pk"
    file_path = os.path.join(filepath, file_name)

    # Save the variable using pickle
    with open(file_path, 'wb') as f:
        pk.dump(variable, f)
        logger.info(f"Variable saved as '{file_path}'.")


def evaluate_model(model_path, attrs, test_features, test_labels, polynomial, ID, path_to_save,
                   model_type):
    """
    Evaluate a model: load weights, make predictions, and calculate moments.

    Args:
        model_path (str): Path to the model state file.
        attrs (dict): Model attributes.
        model_instance (BaseModel): An instance of the model class (e.g., ResNet, PINN).
        test_features (Tensor): Test features as a PyTorch tensor.
        test_labels (Tensor): Test labels as a PyTorch tensor.
        polynomial (int): Polynomial order for moments calculation.
        ID (str): Identifier for saved results.
        path_to_save (str): Directory to save evaluation results.
        model_type (str): Model type (e.g., ResNet, PINN).

    Returns:
        tuple: (moment_error, moment_std)
    """
    logger.info(f"Loading model from {model_path}")
    model_instance = load_model_instance(model_path, attrs, model_type)

    logger.info("Running predictions on test data")
    pred_l = model_instance.forward(test_features)

    moments_act = calc_moments(test_features.numpy(), test_labels.numpy(), polynomial=polynomial)
    moments_pred = calc_moments(test_features.numpy(), pred_l.detach().numpy(), polynomial=polynomial)

    moment_error = np.mean(abs(moments_pred - moments_act), axis=0)
    moment_std = np.std(abs(moments_pred - moments_act), axis=0)

    logger.info(f"Moment error: {moment_error}")
    logger.info(f"Moment standard deviation: {moment_std}")

    save_variable_with_pickle(moment_error, "moment_error", ID, path_to_save)
    save_variable_with_pickle(moment_std, "moment_std", ID, path_to_save)

    return moment_error, moment_std

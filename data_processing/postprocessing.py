import numpy as np
import logging
import os
from models.SaveNLoad import *
import pickle as pk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def evaluate_model(test_features,
                   test_labels,
                   polynomial,
                   model_ID,
                   path_to_save,
                   model_type):
    """
    Evaluate a model: load weights, make predictions, and calculate moments.

    Args:
        test_features (Tensor): Test features as a PyTorch tensor.
        test_labels (Tensor): Test labels as a PyTorch tensor.
        polynomial (int): Polynomial order for moments calculation.
        model_ID (str): Identifier for saved results.
        path_to_save (str): Directory to save evaluation results.
        model_type (str): Model type (e.g., ResNet, PINN).

    Returns:
        tuple: (moment_error, moment_std)
    """

    # Load attributes and evaluate model
    attrs_path = os.path.join(path_to_save, f'attrs{model_ID}.pk')
    model_path = os.path.join(path_to_save, f'{model_type}{model_ID}.pth')
    with open(attrs_path, 'rb') as f:
        attrs = pk.load(f)


    logger.info(f"Loading model from {model_path}")
    model_instance = load_model_instance(model_path, attrs, model_type)

    logger.info("Running predictions on test data")
    model_instance.eval()
    pred_l = model_instance(test_features)

    moments_act = calc_moments(test_features.numpy(), test_labels.numpy(), polynomial=polynomial)
    moments_pred = calc_moments(test_features.numpy(), pred_l.detach().numpy(), polynomial=polynomial)

    moment_error = np.mean(abs(moments_pred - moments_act), axis=0)
    moment_std = np.std(abs(moments_pred - moments_act), axis=0)

    logger.info(f"Moment error: {moment_error}")
    logger.info(f"Moment standard deviation: {moment_std}")

    save_variable_with_pickle(moment_error, "moment_error", model_ID, path_to_save)
    save_variable_with_pickle(moment_std, "moment_std", model_ID, path_to_save)

    return moment_error, moment_std

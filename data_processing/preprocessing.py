import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def import_stored_data(base_path, file, order, noise):
    order_noise_path = base_path

    amat_path = os.path.join(order_noise_path, 'amat', f'amat_{file}.csv')
    psi_path = os.path.join(order_noise_path, 'psi', 'laplace', f'psi_{file}.csv')
    h_path = os.path.join(order_noise_path, 'h', f'h{file}.csv')

    amat = np.genfromtxt(amat_path, delimiter=',', skip_header=0)
    psi = np.genfromtxt(psi_path, delimiter=',', skip_header=0)
    h = np.genfromtxt(h_path, delimiter=',', skip_header=0)

    return amat, psi, h[:-1]  # Remove last element for Fortran hovdx variable


def preprocess_data(path_to_data, file_details, derivative, polynomial, tt_split=0.9, seed=1):
    """
    Preprocess data: import, standardize, split into train, validation, and test sets.

    Args:
        path_to_data (str): Path to the data directory.
        file_details (list of tuples): List of (file_number, noise) tuples.
        derivative (str): Derivative type (e.g., 'Laplace').
        polynomial (int): Polynomial order for moments calculation.
        tt_split (float): Train-test split ratio.
        seed (int): Seed for random splitting.

    Returns:
        tuple: (train_features, train_labels, val_features, val_labels, test_features, test_labels)
    """
    logger.info(f"Starting data preprocessing with file details: {file_details}")

    amat_list = []
    psi_list = []

    for file_number, noise in file_details:
        logger.info(f'Processing file number: {file_number}, noise: {noise}')
        amat, psi, h = import_stored_data(path_to_data, file_number, order=polynomial, noise=noise)
        stand_psi = standardize_psi(psi, h, derivative)
        amat_list.append(amat)
        psi_list.append(stand_psi)

    stand_feature = np.concatenate(amat_list, axis=0)
    stand_label = np.concatenate(psi_list, axis=0)

    train_f, train_l, test_f, test_l, _, _ = create_train_test(stand_feature, stand_label, tt_split=tt_split, seed=seed)
    train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=0.2, random_state=seed)

    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_labels = torch.tensor(train_l, dtype=torch.float32)
    val_features = torch.tensor(val_f, dtype=torch.float32)
    val_labels = torch.tensor(val_l, dtype=torch.float32)
    test_features = torch.tensor(test_f, dtype=torch.float32)
    test_labels = torch.tensor(test_l, dtype=torch.float32)

    logger.info("Data preprocessing complete")
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

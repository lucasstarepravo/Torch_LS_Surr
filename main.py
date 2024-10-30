import torch.optim
from sklearn.model_selection import train_test_split
from data_processing.preprocessing import *
from data_processing.postprocessing import *
from Plots import *
from models.ANN import ANN
from models.PINN import PINN
from models.GP import GP
from models.PINN_ResNet import PINN_ResNet
from models.ResNet import ResNet
from models.SaveNLoad import *
import numpy as np
import pickle as pk
import os
import logging
import torch

import torch.multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_stored_data(base_path, file, order, noise):
    order_noise_path = base_path

    amat_path = os.path.join(order_noise_path, 'amat', f'amat_{file}.csv')
    psi_path = os.path.join(order_noise_path, 'psi', 'laplace', f'psi_{file}.csv')
    h_path = os.path.join(order_noise_path, 'h', f'h{file}.csv')

    amat = np.genfromtxt(amat_path, delimiter=',', skip_header=0)
    psi = np.genfromtxt(psi_path, delimiter=',', skip_header=0)
    h = np.genfromtxt(h_path, delimiter=',', skip_header=0)

    return amat, psi, h[:-1]  # Remove last element for Fortran hovdx variable


def run_model(path_to_data, layers, ID, nprocs, model_type, path_to_save='./data_out'):
    logger.info(f'Running model with layers: {layers} and ID: {ID}')

    # Define file details in a list of tuples (file_number, noise)
    file_details = [(4, 0.3)]

    # Initialize empty lists to store processed data
    amat_list = []
    psi_list = []

    # Derivative and polynomial details
    derivative = 'Laplace'
    polynomial = 2

    # Loop through file details to import and process data
    for file_number, noise in file_details:
        logger.info(f'Processing file number: {file_number}, noise: {noise}')

        # Import data
        amat, psi, h = import_stored_data(path_to_data, file_number, order=polynomial, noise=noise)

        # Standardize psi data
        stand_psi = standardize_psi(psi, h, derivative)

        # Append processed data to lists
        amat_list.append(amat)
        psi_list.append(stand_psi)

    # Concatenate lists to form final datasets
    stand_feature = np.concatenate(amat_list, axis=0)
    stand_label = np.concatenate(psi_list, axis=0)

    # Split data into training and test sets
    train_f, train_l, test_f, test_l, train_index, test_index = create_train_test(stand_feature, stand_label,
                                                                                  tt_split=0.9, seed=1)
    train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=0.2, random_state=1)

    # Convert data to PyTorch tensors
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_labels = torch.tensor(train_l, dtype=torch.float32)
    val_features = torch.tensor(val_f, dtype=torch.float32)
    val_labels = torch.tensor(val_l, dtype=torch.float32)


    ann = PINN(layers,
                 optimizer='adam',
                 loss_function='MSE', epochs=8, batch_size=64, train_f=train_features,
                 train_l=train_labels, val_f=val_features, val_l=val_labels, moments=polynomial, final_alpha=0.5)
    #ann.fit()

    # ann = ANN(layers, optimizer='adam', loss_function='MSE',
    #          epochs=1500, batch_size=64, train_f=train_features, train_l=train_labels, val_f=val_features,
    #          val_l=val_labels)


    #ann = ResNet(layers, optimizer='adam',
    #              loss_function='MSE', epochs=1000, batch_size=64, train_f=train_features, train_l=train_labels,
    #              val_f=val_features, val_l=val_labels, skip_connections=[(0, 99), (1, 19), (20, 39), (40, 59),
    #              (60, 79), (80, 98)])

    # pinn_res2 = PINN_ResNet(layers,
    #                       optimizer='adam', loss_function='sgs', epochs=3000, batch_size=128, train_f=train_features,
    #                       train_l=train_labels, val_f=val_features, val_l=val_labels,
    #                       skip_connections=[(0, 99), (1, 19), (20, 39), (40, 59), (60, 79), (80, 98)],
    #                       moments=polynomial, final_alpha=0.5, alpha_epoch_start=1000)

    # Initialize and train model
    #ann = ANN(layers, optimizer='adam', loss_function='MSE', epochs=1500, batch_size=64,
    #          train_f=train_features, train_l=train_labels, val_f=val_features, val_l=val_labels)

    logger.info('Starting model training')
    mp.spawn(ann.fit, args=(nprocs, path_to_save, 'pinn', ID), nprocs=nprocs)

    # Loading saved model
    attrs_path = os.path.join(path_to_save, f'attrs{ID}.pk')
    model_path = os.path.join(path_to_save, f'{model_type}{ID}.pth')
    attr = load_attrs(attrs_path)
    ann = load_model_instance(model_path, attr, model_type)

    # Predict on test data
    test_features = torch.tensor(test_f, dtype=torch.float32)
    pred_l = ann.predict(test_features)

    # Calculate moments and moment errors
    moments_act = calc_moments(test_f, test_l, polynomial=polynomial)
    moments_pred = calc_moments(test_f, pred_l.detach().numpy(), polynomial=polynomial)
    moment_error = np.mean(abs(moments_pred - moments_act), axis=0)
    moment_std = np.std(abs(moments_pred - moments_act), axis=0)

    # Save variables
    save_variable_with_pickle(moment_error, "moment_error", ID, path_to_save)
    save_variable_with_pickle(moment_std, "moment_std", ID, path_to_save)

    logger.info('Model run complete')


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


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_model('/mnt/iusers01/mace01/w32040lg/mfree_surr/data/Order_2/Noise_0.3/Data', 7*[64], ID='66',
              nprocs=2, path_to_save='./data_out', model_type='pinn')

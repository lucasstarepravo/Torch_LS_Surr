from data_processing.preprocessing import preprocess_data
from data_processing.postprocessing import evaluate_model
from Plots import *
from models.NN_Base import BaseModel
from models.PINN import PINN
from models.ResNet import ResNet
import pickle as pk
import os
import logging
import torch.multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_model(path_to_data, layers, model_ID, nprocs, model_type, file_details, path_to_save='./data_out'):
    logger.info(f'Running model with layers: {layers} and ID: {model_ID}')

    # File and preprocessing details

    derivative = 'Laplace'
    polynomial = 2

    # Preprocess data
    (train_features,
     train_labels,
     val_features,
     val_labels,
     test_features,
     test_labels) = preprocess_data(path_to_data, file_details, derivative, polynomial)

    skip_connections = [(0, 9)]

    ann = PINN(hidden_layers=layers,
               optimizer='adam',
               loss_function='MSE',
               epochs=100000,
               batch_size=128,
               train_f=train_features,
               train_l=train_labels,
               moments_order=polynomial,
               alpha=0.5)

    logger.info('Starting model training')
    mp.spawn(ann.fit,
             args=(nprocs, path_to_save, model_type, model_ID, train_features, train_labels, val_features, val_labels),
             nprocs=nprocs)

    evaluate_model(test_features, test_labels, polynomial, model_ID, path_to_save, model_type)

    logger.info('Model run complete')



if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_model('/mnt/iusers01/mace01/w32040lg/mfree_surr/data/Order_2/Noise_0.3/Data2',
              layers=7 * [64],
              model_ID='66',
              nprocs=2,
              model_type='pinn',
              file_details=[(8, 0.3)],
              path_to_save='./data_out')

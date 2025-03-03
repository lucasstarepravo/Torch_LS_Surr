from data_processing.preprocessing import preprocess_data
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

    if model_type.lower() == 'transformers':
        train_features = train_features.reshape(train_features.shape[0], -1, 5)
        val_features = val_features.reshape(val_features.shape[0], -1, 5)
        test_features = test_features.reshape(test_features.shape[0], -1, 5)

        #train_labels = train_labels.unsqueeze(-1)
        #val_labels = val_labels.unsqueeze(-1)
        #test_labels = np.expand_dims(test_labels, axis=-1)

    ann = ResNet(hidden_layers=layers,
                 optimizer='adam',
                 loss_function='MSE',
                 epochs=13,
                 batch_size=128,
                 train_f=train_features,
                 train_l=train_labels,
                 skip_connections=skip_connections)

    logger.info('Starting model training')
    mp.spawn(ann.fit,
             args=(nprocs, path_to_save, model_type, model_ID,
                   train_features, train_labels, val_features, val_labels, None,
                   test_features, test_labels, polynomial),
             nprocs=nprocs)

    logger.info('Model run complete')


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_model(path_to_data='/home/w32040lg/Shape Function Surrogate/Data2',
              layers=7 * [64],
              model_ID='777',
              nprocs=2,
              model_type='transformers',
              file_details=[(7, 0.3)],
              path_to_save='./data_out')

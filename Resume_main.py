import os
from models.Resume_fn import initialise_instance
from data_processing.preprocessing import preprocess_data
import torch.multiprocessing as mp
from data_processing.postprocessing import evaluate_model
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # File and preprocessing details
    file_details = [(8, 0.3)]
    derivative = 'Laplace'
    polynomial = 2
    path_to_data = '/mnt/iusers01/mace01/w32040lg/mfree_surr/data/Order_2/Noise_0.3/Data2'

    nprocs = 2
    path_to_save = './data_out'
    model_type = 'pinn'
    model_ID = 777
    epochs = 12

    # Preprocess data
    train_f, train_l, val_f, val_l, test_features, test_labels = preprocess_data(
        path_to_data, file_details, derivative, polynomial)
    model_instance, optimiser_state = initialise_instance(
                      path_to_save=path_to_save,
                      model_type=model_type,
                      model_ID=model_ID,
                      epochs=epochs)

    mp.spawn(
        model_instance.fit,
        args=(nprocs, path_to_save, model_type, model_ID, train_f, train_l, val_f, val_l, optimiser_state,
              test_features, test_labels, polynomial),
        nprocs=nprocs,
        join=True)

    logger.info('Model run complete')
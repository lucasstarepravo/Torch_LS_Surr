import os
from models.Resume_fn import initialise_instance
from data_processing.preprocessing import preprocess_data
import torch.multiprocessing as mp

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

    # Preprocess data
    train_f, train_l, val_f, val_l, test_features, test_labels = preprocess_data(
        path_to_data, file_details, derivative, polynomial)
    model_instance = initialise_instance(
                      path_to_save='./data_out',
                      model_type='pinn',
                      model_ID=777,
                      epochs=12)

    mp.spawn(
        model_instance.fit,
        args=(nprocs, path_to_save, model_type, model_ID, train_f, train_l, val_f, val_l, True),
        nprocs=nprocs,
        join=True)

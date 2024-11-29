import os
from models.NN_Base import BaseModel
from data_processing.preprocessing import preprocess_data

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # File and preprocessing details
    file_details = [(8, 0.3)]
    derivative = 'Laplace'
    polynomial = 2
    path_to_data = '/mnt/iusers01/mace01/w32040lg/mfree_surr/data/Order_2/Noise_0.3/Data2'

    # Preprocess data
    train_features, train_labels, val_features, val_labels, test_features, test_labels = preprocess_data(
        path_to_data, file_details, derivative, polynomial)
    BaseModel.continue_training(nprocs=2,
                                path_to_save='./data_out',
                                model_type='pinn',
                                model_ID='63',
                                epochs=1000,
                                train_f=train_features,
                                train_l=train_labels,
                                val_f=val_features,
                                val_l=val_labels)


# Will have to change how this is called now that I separated the topology and wrapper class
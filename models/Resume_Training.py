import os
from models.NN_Base import BaseModel

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    BaseModel.continue_training(nprocs=2,
                                path_to_save='./data_out',
                                model_type='pinn',
                                model_ID='63',
                                instance_path='./data_out/checkpoint',
                                epochs=10000)

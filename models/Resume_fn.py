import os
from models.NN_Base import BaseModel
from models.ResNet import ResNet
from models.PINN import PINN
import torch
from torch import Tensor
import pickle as pk
import torch.multiprocessing as mp
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present



def continue_training(nprocs, path_to_save, model_type, model_ID, epochs,
                      train_f: Tensor,
                      train_l: Tensor,
                      val_f: Tensor,
                      val_l: Tensor):

    attr_path = os.path.join(path_to_save, f'checkpoint_attrs{model_ID}.pk')
    with open(attr_path, 'rb') as f:
        attrs = pk.load(f)

    if model_type.lower() == 'ann':
        model_instance = BaseModel(
            hidden_layers=attrs['hidden_layers'],
            optimizer=attrs['optimizer_str'],
            loss_function=attrs['loss_function'],
            epochs=epochs,
            batch_size=attrs['batch_size'],
            train_f=attrs['input_size'],
            train_l=attrs['output_size'])

    elif model_type.lower() == 'pinn':
        model_instance = PINN(
            alpha=attrs['alpha'],
            moments_order=attrs['moments_order'],
            hidden_layers=attrs['hidden_layers'],
            optimizer=attrs['optimizer_str'],
            loss_function=attrs['loss_function'],
            epochs=epochs,
            batch_size=attrs['batch_size'],
            train_f=attrs['input_size'],
            train_l=attrs['output_size'])

    elif model_type.lower() == 'resnet':
        model_instance = ResNet(
            skip_connections=attrs['skip_connections'],
            hidden_layers=attrs['hidden_layers'],
            optimizer=attrs['optimizer_str'],
            loss_function=attrs['loss_function'],
            epochs=epochs,
            batch_size=attrs['batch_size'],
            train_f=attrs['input_size'],
            train_l=attrs['output_size'])
    else:
        raise ValueError('Model type not supported')

    # give training and validation losses to instance
    model_instance.tr_loss = attrs['tr_loss']
    model_instance.val_loss = attrs['val_loss']
    model_instance.best_val_loss = attrs['best_val_loss']

    # load optimizer state
    optimizer_state = attrs['optimizer']
    model_instance.optimizer.load_state_dict(optimizer_state)

    # load checkpoint weights state
    state_path = os.path.join(path_to_save, f"checkpoint_{model_type}{model_ID}.pth")
    model_state = torch.load(state_path, map_location=torch.device('cpu'))
    consume_prefix_in_state_dict_if_present(model_state, prefix="module.")
    model_instance.model.load_state_dict(model_state)
    model_instance.best_model_wts = model_state

    mp.spawn(
        model_instance.fit,
        args=(nprocs, path_to_save, model_type, model_ID, train_f, train_l, val_f, val_l, True),
        nprocs=nprocs,
        join=True)

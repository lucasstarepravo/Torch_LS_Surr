import torch
from models.ANN import ANN_topology
from models.PINN import PINN_topology
import pickle as pk
import os


def save_model_instance(ann_instance, filepath, model_type, model_ID):
    # Ensure the directory exists
    os.makedirs(filepath, exist_ok=True)

    # Save the PyTorch model's state dict
    model_path = os.path.join(filepath, f'{model_type}{model_ID}.pth')
    torch.save(ann_instance.state_dict(), model_path)

    # Prepare attributes to save
    attrs = {
        'input_size': ann_instance.input_size,
        'output_size': ann_instance.output_size,
        'hidden_layers': ann_instance.hidden_layers,
        'history': (ann_instance.training_loss, ann_instance.val_loss)
    }

    # Save the attributes using pickle
    attrs_path = os.path.join(filepath, f'attrs{model_ID}.pk')
    with open(attrs_path, 'wb') as f:
        pk.dump(attrs, f)


def load_attrs(filepath):
    ''' In this case the filepath must contain the complete directory including the file'''

    with open(filepath, 'rb') as f:
        attrs = pk.load(f)
    return attrs


def load_model_instance(filepath, attrs, model_type='ANN'):
    ''' In this case the filepath must contain the complete directory including the file'''
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']
    model_state = torch.load(filepath)

    # Load the appropriate model type
    if model_type == 'ANN':
        model = ANN_topology(input_size, output_size, hidden_layers)
    elif model_type == 'PINN':
        model = PINN_topology(input_size, output_size, hidden_layers)
    else:
        raise ValueError("Invalid model_type. Must be 'ANN' or 'PINN'.")

    # Load the saved state dict into the model
    model.load_state_dict(model_state)

    return model

import torch
from models.ANN import ANN_topology
from models.PINN import PINN_topology
import pickle as pk


def save_model_instance(ann_instance, filepath, model_type, model_ID):
    # Save the PyTorch model's state dict
    torch.save(ann_instance.model.state_dict(), filepath + '/' + str(model_type) + str(model_ID)+".pth")

    attrs = {
        'input_size': ann_instance.input_size,
        'output_size': ann_instance.output_size,
        'hidden_layers': ann_instance.hidden_layers,
        'history': (ann_instance.training_loss, ann_instance.val_loss)
    }

    with open(filepath + '/attrs' + str(model_ID) + '.pk', 'wb') as f:
        pk.dump(attrs, f)


def load_attrs(filepath):
    ''' In this case the filepath must contain the complete directory including the file'''

    with open(filepath, 'rb') as f:
        attrs = pk.load(f)
    return attrs


def load_model_instance(filepath, attrs):
    ''' In this case the filepath must contain the complete directory including the file'''
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']
    model_state = torch.load(filepath)

    # In this case it doesn't matter if ANN_topology or PINN_topology is used as they
    model = ANN_topology(input_size, output_size, hidden_layers)
    model.load_state_dict(model_state)
    return model

import torch
from models.NN_Base import NN_Topology
from models.ResNet import ResNet_Topology
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def load_model_instance(model_path,
                        attrs,
                        model_type):
    """
    Args:
        model_path (str): Path to the saved model's state_dict file.
        attrs (dict): Attributes of the model (e.g., input_size, output_size).
        model_type (str): Type of the model (e.g., ResNet, ResNet, etc.).

    Returns:
        BaseModel: A model instance with the loaded weights and attributes.
    """
    # Initialize the model using its attributes
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']

    model_state = torch.load(model_path, map_location=torch.device('cpu'))

    # In the case of ResNet skip_connections must be obtained too
    if model_type.lower() == 'ann' or model_type.lower() == 'pinn':
        model_instance = NN_Topology(input_size, hidden_layers, output_size)
    elif model_type.lower() == 'resnet':
        skip_connections = attrs['skip_connections']
        model_instance = ResNet_Topology(input_size, hidden_layers, output_size, skip_connections)
    else:
        raise ValueError('model_type must be one of "ann","pinn","resnet"')

    # Automatically remove the "module." prefix if it exists
    consume_prefix_in_state_dict_if_present(model_state, prefix="module.")
    model_instance.load_state_dict(model_state)

    return model_instance


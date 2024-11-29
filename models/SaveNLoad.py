import torch
from models.NN_Base import NN_Topology
from models.ResNet1 import ResNet_Topology


def load_model_instance(filepath, attrs, model_type):
    """
    Load a model instance with saved attributes and weights.

    Args:
        filepath (str): Path to the saved model's state_dict file.
        attrs (dict): Attributes of the model (e.g., input_size, output_size).
        model_instance (BaseModel): An instance of the model class (e.g., ResNet, PINN, etc.).
        model_type (str): Type of the model (e.g., ResNet, ResNet, etc.).

    Returns:
        BaseModel: A model instance with the loaded weights and attributes.
    """
    if model_type.lower() not in ['ann', 'pinn', 'resnet']:
        raise ValueError('model_type must be one of "ann","pinn","resnet"')

    # Initialize the model using its attributes
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']

    model_state = torch.load(filepath, map_location=torch.device('cpu'))

    # In the case of ResNet skip_connections must be obtained too
    if model_type.lower() == 'ann' or model_type.lower() == 'pinn':
        model = NN_Topology(input_size, output_size, hidden_layers)
    elif model_type.lower() == 'resnet':
        skip_connections = attrs['skip_connections']
        model = ResNet_Topology(input_size, hidden_layers, output_size, skip_connections)

    # Remove the 'module.' prefix from the state dict if present
    state_dict = model_state
    # Automatically remove the "module." prefix if it exists
    if any(key.startswith("module.") for key in state_dict.keys()):
        model.load_state_dict(torch.nn.Module.consume_prefix_in_state_dict_if_present(state_dict, prefix="module."))
    else:
        model.load_state_dict(state_dict)

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    return model


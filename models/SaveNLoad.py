import torch


def load_model_instance(filepath, attrs, model_instance):
    """
    Load a model instance with saved attributes and weights.

    Args:
        filepath (str): Path to the saved model's state_dict file.
        attrs (dict): Attributes of the model (e.g., input_size, output_size).
        model_instance (BaseModel): An instance of the model class (e.g., ResNet, PINN, etc.).

    Returns:
        BaseModel: A model instance with the loaded weights and attributes.
    """
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']
    model_state = torch.load(filepath, map_location=torch.device('cpu'))


    # Initialize the model using its attributes
    model_instance.input_size = input_size
    model_instance.output_size = output_size
    model_instance.hidden_layers = hidden_layers
    model_instance.model = model_instance.create_model()  # Create the model using the child class logic

    # Remove the 'module.' prefix from the state dict if present
    state_dict = model_state
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    # Load the state dict into the model
    model_instance.model.load_state_dict(state_dict)

    return model_instance


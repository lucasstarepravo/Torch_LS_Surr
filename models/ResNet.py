import torch.nn as nn
from torch import Tensor
import torch
from models.NN_Base import BaseModel
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResNet_Topology(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, skip_connections):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.skip_connections = skip_connections if skip_connections else []

        # Scaled skip connection indices
        self.scaled_skip_connection = [(x * 2 + 1, y * 2 + 1) for x, y in self.skip_connections]

        for skip_in, skip_out in skip_connections:
            if skip_in < 0 or skip_in > len(hidden_layers) or skip_out < 0 or skip_out > len(hidden_layers):
                raise IndexError("Invalid skip connection: indices must be within the range of hidden layers.")

        # Define the layers
        layers = [nn.Linear(self.input_size, self.hidden_layers[0])]
        layers += [nn.LayerNorm(self.hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(nn.LayerNorm(self.hidden_layers[i]))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = {}

        for i, layer in enumerate(self.layers):
            # Process each layer based on its type
            if isinstance(layer, nn.Linear):
                x = layer(x)

            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
                outputs[i] = x  # Store the output for potential skip connections

            elif isinstance(layer, nn.SiLU):
                x = layer(x)

            # Apply skip connections dynamically
            for skip_in, skip_out in self.scaled_skip_connection:
                if i == skip_out:
                    if skip_in in outputs:
                        x = x + outputs[skip_in]  # Add skip connection output
                    else:
                        logger.warning(
                            f"Skip connection from {skip_in} to {skip_out} cannot be applied. Key {skip_in} missing.")

        return x



class ResNet(BaseModel):
    """
    skip_connections[list]: must be a list of boolean, the first entry of each boolean is the start of the residual
    block and the second entry is the end of the residual block.
    """
    def __init__(self,
                 skip_connections: list,
                 hidden_layers: list,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor | int,
                 train_l: Tensor | int) -> None:

        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l)
        self.skip_connections = skip_connections
        self.model = ResNet_Topology(self.input_size, hidden_layers, self.output_size, skip_connections)


    def save_checkpoint(self, path_to_save, model_type, model_ID, model_ddp, **kwargs):
        extra_attrs = {'skip_connections': self.skip_connections}
        super().save_checkpoint(path_to_save, model_type, model_ID, model_ddp, **extra_attrs)


    def save_model(self, path_to_save: str, model_type: str, model_ID: str, **kwargs):
        """Save the best model weights with additional attributes specific to ResNet."""
        # Additional attributes specific to this class
        extra_attrs = {'skip_connections': self.skip_connections}
        # Call the parent method with additional attributes
        super().save_model(path_to_save, model_type, model_ID, **extra_attrs)


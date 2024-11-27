import torch.nn as nn
from torch import Tensor
from models.NN_Base import BaseModel
import torch


def calc_moments_torch(stand_f, pred_l, n):
    """ Calculate right-hand side vector of the linear system with the predicted solution vector """
    stand_f = stand_f.view(stand_f.shape[0], n, n)
    pred_l = pred_l.unsqueeze(-1)
    moments = torch.matmul(stand_f, pred_l)
    return moments.squeeze(-1)


class PINN(BaseModel):
    def __init__(self,
                 alpha: float,
                 moments_order: int,
                 hidden_layers: list,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor,
                 val_f: Tensor,
                 val_l: Tensor) -> None:

        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l, val_f, val_l)
        self.alpha = alpha
        self.moments_order = int(moments_order)

    def physics_loss_fn(self, outputs, inputs):
        n = int((self.moments_order ** 2 + 3 * self.moments_order) / 2)
        moments = calc_moments_torch(inputs, outputs, n)
        target_moments = torch.zeros((outputs.shape[0], n), device=outputs.device)
        target_moments[:, 2] = 1
        target_moments[:, 4] = 1
        physics_loss = (target_moments - moments) ** 2
        return physics_loss.mean(axis=0)

    def calculate_loss(self, outputs, labels, inputs=None):
        if inputs is None:
            raise ValueError("Inputs cannot be None for this child class.")
        physics_loss = self.physics_loss_fn(outputs, inputs)
        data_loss = self.loss_function(outputs, labels)
        return (1 - self.alpha) * data_loss + self.alpha * physics_loss.mean()


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
                 train_f: Tensor,
                 train_l: Tensor,
                 val_f: Tensor,
                 val_l: Tensor) -> None:

        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l, val_f, val_l)
        self.skip_connections = skip_connections

    def create_model(self):
        self.skip_connections = self.skip_connections if self.skip_connections else []
        self.scaled_skip_connection = [(x * 2 + 1, y * 2 + 1) for x, y in self.skip_connections]

        layers = [nn.Linear(self.input_size, self.hidden_layers[0])]
        layers += [nn.LayerNorm(self.hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(nn.LayerNorm(self.hidden_layers[i]))
            layers.append(nn.SiLU())

        # Add the final layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        # Use ModuleList to hold all the layers
        return nn.ModuleList(layers)

    def forward(self, inputs): # make sure the predict method of the parent class is using this forward propagation
        resblock_output = {}
        for i, layer in enumerate(self.model): # self.model was initially self.layers, which is originally the output of the method above
            res_block = [(index, tup) for index, tup in enumerate(self.scaled_skip_connection)
                         if i in tup]  # This checks if the current layer is supposed to be a residual block
            # For now, this code only works with unique receiving and sending block layers (a layer cannot receive to past outputs, or receive and send the output through the same layer)
            if res_block:
                if i == res_block[0][1][0]:  # This checks if this is the start or finish of residual block
                    inputs = layer(inputs)
                    resblock_output[res_block[0][1][1]] = inputs
                else:
                    inputs = layer(inputs)
                    inputs = inputs + resblock_output[i]
            else:
                inputs = layer(inputs)
        return inputs

    def save_model(self, path_to_save: str, model_type: str, model_ID: str, **kwargs):
        """Save the best model weights with additional attributes specific to ResNet."""
        # Additional attributes specific to this class
        extra_attrs = {'skip_connections': self.skip_connections}
        # Call the parent method with additional attributes
        super().save_model(path_to_save, model_type, model_ID, **extra_attrs)


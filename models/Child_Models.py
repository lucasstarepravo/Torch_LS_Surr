import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.NN_Base import BaseModel
import torch
import time
from tqdm import tqdm


def monomial_power_torch(polynomial):
    monomial_exponent = []
    for total_polynomial in range(1, polynomial + 1):
        for i in range(total_polynomial + 1):
            monomial_exponent.append((total_polynomial - i, i))
    # Convert list of tuples to a PyTorch tensor
    return torch.tensor(monomial_exponent, dtype=torch.int)


def calc_moments_torch(stand_f, pred_l, n):
    stand_f = stand_f.view(stand_f.shape[0], n, n)
    pred_l = pred_l.unsqueeze(-1)
    moments = torch.matmul(stand_f, pred_l)
    return moments.squeeze(-1)


class PINN(BaseModel):
    def __init__(self, alpha, moments_order, hidden_layers, optimizer, loss_function, epochs, batch_size, train_f,
                 train_l, val_f, val_l):
        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l, val_f, val_l)
        self.alpha = alpha
        self.moments_order = int(moments_order)

    def physics_loss_fn(self, outputs, inputs):
        n = int((self.moments_order ** 2 + 3 * self.moments_order) / 2)
        moments = calc_moments_torch(inputs, outputs, n)

        # target_moments = torch.zeros((outputs.shape[0], n)) #make sure the outputs.shape[0] is capturing the dimension I want
        # Create the target moments tensor on the same device as the outputs
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
    def __init__(self, skip_connections, hidden_layers, optimizer, loss_function, epochs, batch_size, train_f,
                 train_l, val_f, val_l):
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

    def forward(self, x): # make sure the predict method of the parent class is using this forward propagation
        resblock_output = {}
        for i, layer in enumerate(self.model): # self.model was initially self.layers, which is originally the output of the method above
            res_block = [(index, tup) for index, tup in enumerate(self.scaled_skip_connection) if
                         i in tup]  # This checks if the current layer is supposed to be a residual block
            # For now, this code only works with unique receiving and sending block layers (a layer cannot receive to past outputs, or receive and send the output through the same layer)
            if res_block:
                if i == res_block[0][1][0]:  # This checks if this is the start or finish of residual block
                    x = layer(x)
                    resblock_output[res_block[0][1][1]] = x
                else:
                    x = layer(x)
                    x = x + resblock_output[i]
            else:
                x = layer(x)
        return x
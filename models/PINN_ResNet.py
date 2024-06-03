import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import math


class PINN_ResNet_topology(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, skip_connections):
        super(PINN_ResNet_topology, self).__init__()
        self.skip_connections = skip_connections if skip_connections else []
        self.scaled_skip_connection = [(x * 2 + 1, y * 2 + 1) for x, y in skip_connections]

        layers = [nn.Linear(input_size, hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.SiLU())

        # Add the final layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Use ModuleList to hold all the layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        resblock_output = {}

        for i, layer in enumerate(self.layers):

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


def calculate_val_loss(model, val_loader, loss_function):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No gradients required for validation step
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)


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


def physics_loss_fn(outputs, inputs, physics_loss):
    n = int((physics_loss ** 2 + 3 * physics_loss) / 2)
    moments = calc_moments_torch(inputs, outputs, n)

    target_moments = torch.zeros(
        (outputs.shape[0], n))  # make sure the outputs.shape[0] is capturing the dimension I want
    target_moments[:, 2] = 1
    target_moments[:, 4] = 1
    physics_loss = (target_moments - moments) ** 2
    return physics_loss.mean(axis=0)


class PINN_ResNet:
    def __init__(self, hidden_layers, optimizer, loss_function, epochs, batch_size, train_f,
                 train_l, val_f, val_l, moments, dynamic_physics_loss=False,
                 alpha_epoch_start=0, alpha_epoch_stop=None, final_alpha=0.5, skip_connections=None):
        '''alpha_epoch_stop must be bigger than alpha_epoch_start'''
        self.input_size = train_f.shape[1]
        self.output_size = train_l.shape[1]
        self.hidden_layers = hidden_layers
        self.model = PINN_ResNet_topology(self.input_size, self.output_size, hidden_layers, skip_connections)
        self.optimizer = self.define_optimizer(optimizer)
        self.loss_function = self.define_loss(loss_function)
        self.epochs = epochs
        self.train_loader = DataLoader(TensorDataset(train_f, train_l), batch_size=batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(TensorDataset(val_f, val_l), batch_size=batch_size,
                                     shuffle=False)
        self.best_model_wts = None
        self.training_loss = None
        self.val_loss = None
        self.batch_size = batch_size
        self.moments = int(moments)
        self.final_alpha = final_alpha
        self.alpha_epoch_start = alpha_epoch_start

        # Checking if there will be a dynamic increase of the loss weight
        if dynamic_physics_loss:
            # If yes, self.dynamic_physics_loss is True and the start of the weight is optional, the standard option
            # starts implementing the weight in the first epoch
            self.dynamic_physics_loss = dynamic_physics_loss
        else:
            self.dynamic_physics_loss = False

        # If alpha_epoch_stop is given any value, this value will become the final weight increment of the physics loss
        if alpha_epoch_stop is not None:
            self.alpha_epoch_stop = alpha_epoch_stop
        else:
            # If alpha is not given a value, it will be only finished incremented in the last epoch
            self.alpha_epoch_stop = epochs

    def define_optimizer(self, optimiser, lr=0.01, momentum=0.9):
        if optimiser == 'adam':
            return torch.optim.Adam(self.model.parameters())
        elif optimiser == 'sgs':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    @staticmethod
    def define_loss(loss_function):
        if loss_function == 'MAE':
            return torch.nn.L1Loss()
        elif loss_function == 'MSE':
            return torch.nn.MSELoss()

    def fit(self):
        best_val_loss = float('inf')
        best_model_wts = None

        training_start_time = time.time()

        training_losses = []
        validation_losses = []

        if self.dynamic_physics_loss:
            alpha = 0
            alpha_increments = (self.final_alpha - alpha) / (self.alpha_epoch_stop - self.alpha_epoch_start)

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            self.model.train()  # Set model to training mode
            running_loss = 0.0

            if self.dynamic_physics_loss:
                if self.alpha_epoch_start <= epoch <= self.alpha_epoch_stop:
                    alpha = alpha + alpha_increments
            else:
                if self.alpha_epoch_start <= epoch:
                    alpha = self.final_alpha
                else:
                    alpha = 0

            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()  # Clear previous gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.loss_function(outputs, labels)  # Calculate loss

                physics_loss = physics_loss_fn(outputs, inputs, self.moments)

                # Below I can implement different weights for the monomial physics loss
                total_loss = (1 - alpha) * loss + alpha * physics_loss.mean()

                total_loss.backward()  # Backward pass
                self.optimizer.step()  # Update labels

                running_loss += loss.item() * inputs.size(0)

            avg_training_loss = running_loss / len(self.train_loader.dataset)
            training_losses.append(avg_training_loss)

            # Calculate validation loss after each epoch
            val_loss = calculate_val_loss(self.model, self.val_loader, self.loss_function)
            validation_losses.append(val_loss)

            # Save the model if the validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = self.model.state_dict().copy()  # Deep copy the model labels

            epoch_time = time.time() - epoch_start_time
            print(
                f'Epoch {epoch + 1}/{self.epochs} - Loss: {avg_training_loss:.4e}, '
                f'Validation Loss: {val_loss:.4e}, Time: {epoch_time:.2f}s')

        self.best_model_wts = best_model_wts

        # Calculate and print the total training time
        total_training_time = time.time() - training_start_time
        print(f'Total training time: {total_training_time:.3f}s')

        self.training_loss = training_losses
        self.val_loss = validation_losses

        # Load best model labels
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)

    def predict(self, input):
        self.model.eval()

        # Split the input_tensor into batches.
        num_samples = input.size(0)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size  # Calculate how many batches are needed.

        predictions = []  # List to store the predictions.

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc='Predicting'):
                # Extract the batch from input_tensor.
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch = input[start_idx:end_idx]

                # Make predictions for the current batch.
                batch_predictions = self.model(batch)
                predictions.append(batch_predictions)

            predictions = torch.cat(predictions, dim=0)
            return predictions

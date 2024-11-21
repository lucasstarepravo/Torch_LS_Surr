from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.distributed import init_process_group
import logging
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present as consume_pref
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import pickle as pk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResNet_topology(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, skip_connections):
        super(ResNet_topology, self).__init__()
        self.skip_connections = skip_connections if skip_connections else []
        self.scaled_skip_connection = [(x * 2 + 1, y * 2 + 1) for x, y in skip_connections]

        layers = [nn.Linear(input_size, hidden_layers[0])]
        layers += [nn.LayerNorm(hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.LayerNorm(hidden_layers[i]))
            layers.append(nn.SiLU())

        # Add the final layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Use ModuleList to hold all the layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        resblock_output = {}

        for i, layer in enumerate(self.layers):

            res_block = [(index, tup) for index, tup in enumerate(self.scaled_skip_connection) if i in tup] # This checks if the current layer is supposed to be a residual block
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

    def predict(self, x):
        x = self.forward(x)
        return x


def calculate_val_loss(model, val_loader, loss_function, proc_index):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No gradients required for validation step
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(proc_index), labels.to(proc_index)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)


def define_loss(loss_function):
    if loss_function == 'MAE':
        return torch.nn.L1Loss()
    elif loss_function == 'MSE':
        return torch.nn.MSELoss()


class ResNet:
    def __init__(self, hidden_layers, optimizer, loss_function, epochs, batch_size, train_f,
                 train_l, val_f, val_l, skip_connections=None):

        self.loss_function = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.input_size = train_f.shape[1]
        self.output_size = train_l.shape[1]
        self.hidden_layers = hidden_layers
        self.optim_choice = optimizer
        self.lossfunc_choice = loss_function
        self.skip_connections = skip_connections


        self.epochs = epochs
        self.train_f = train_f
        self.train_l = train_l
        self.val_f = val_f
        self.val_l = val_l

        self.model = None
        self.best_model_wts = None
        self.training_loss = None
        self.val_loss = None
        self.batch_size = batch_size


    def fit(self, proc_index, nprocs, path_to_save, model_type, model_ID):
        logger.debug('Importing backend')
        # Initialising backend
        # If want to use GPU, use nccl backend
        init_process_group(backend='nccl', world_size=nprocs, rank=proc_index)

        # Enumerating process -- Only necessary for GPU
        torch.cuda.set_device(proc_index)

        logger.debug('Formating data')
        # Preparing data
        train_tensor = TensorDataset(self.train_f, self.train_l)
        tr_sampler = torch.utils.data.distributed.DistributedSampler(train_tensor,
                                                                     num_replicas=nprocs, rank=proc_index)
        self.train_loader = torch.utils.data.DataLoader(train_tensor,
                                                        batch_size=self.batch_size, sampler=tr_sampler, num_workers=0)

        val_tensor = TensorDataset(self.val_f, self.val_l)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_tensor,
                                                                      num_replicas=nprocs, rank=proc_index)
        self.val_loader = torch.utils.data.DataLoader(val_tensor,
                                                      batch_size=self.batch_size, sampler=val_sampler, num_workers=0)

        logger.debug('Initialising model')
        # Initialising model -- append the end if using GPU
        self.model = ResNet_topology(self.input_size, self.output_size, self.hidden_layers,self.skip_connections).to(proc_index)

        # Optimiser
        if self.optim_choice == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())

        # Loss function
        self.loss_function = define_loss(self.lossfunc_choice)

        logger.info('Wrapping model with DDP')
        # model_ddp = DDP(self.model)
        # For GPU
        model_ddp = DDP(self.model, device_ids=[proc_index], output_device=proc_index)


        best_val_loss = float('inf')
        best_model_wts = None

        training_start_time = time.time()

        training_losses = []
        validation_losses = []

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            model_ddp.train()  # Set model to training mode
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                # Move data to the correct GPU
                inputs, labels = inputs.to(proc_index), labels.to(proc_index)

                self.optimizer.zero_grad()  # Clear previous gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.loss_function(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update labels

                running_loss += loss.item() * inputs.size(0)

            avg_training_loss = running_loss / len(self.train_loader.dataset)
            training_losses.append(avg_training_loss)

            # Calculate validation loss after each epoch
            val_loss = calculate_val_loss(self.model, self.val_loader, self.loss_function, proc_index)
            validation_losses.append(val_loss)

            # Save the model if the validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = self.model.state_dict().copy()  # Deep copy the model labels

            epoch_time = time.time() - epoch_start_time
            if proc_index==0:
                print(
                      f'Epoch {epoch + 1}/{self.epochs} - Loss: {avg_training_loss:.4e}, '
                      f'Validation Loss: {val_loss:.4e}, Time: {epoch_time:.2f}s')

        self.best_model_wts = best_model_wts

        # Calculate and print the total training time
        total_training_time = time.time() - training_start_time
        if proc_index == 0:
            print(f'Total training time: {total_training_time:.3f}s')

        self.training_loss = training_losses
        self.val_loss = validation_losses

        # Save model, training history and attributes
        if proc_index == 0 and self.best_model_wts is not None:
            logger.info(f'Process {proc_index} is attempting to save the model.')
            # Ensure the directory exists
            os.makedirs(path_to_save, exist_ok=True)

            # Save the PyTorch model's state dict from the `model` attribute
            model_path = os.path.join(path_to_save, f'{model_type}{model_ID}.pth')
            consume_pref(self.best_model_wts, prefix="module.")

            # Prepare attributes to save
            attrs = {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': self.hidden_layers,
                'history': (self.training_loss, self.val_loss)
            }

            # Save the attributes using pickle
            attrs_path = os.path.join(path_to_save, f'attrs{model_ID}.pk')
            with open(attrs_path, 'wb') as f:
                pk.dump(attrs, f)

            self.model.load_state_dict(self.best_model_wts)
            torch.save(self.model.state_dict(), model_path)  # Access the internal `model` attribute

        dist.destroy_process_group()

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

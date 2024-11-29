import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor
import pickle as pk
import torch.multiprocessing as mp

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def define_loss(loss_function):
    if loss_function == 'MAE':
        return torch.nn.L1Loss()
    elif loss_function == 'MSE':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")


class NN_Topology(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(NN_Topology, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        layers = [nn.Linear(self.input_size, self.hidden_layers[0])]
        layers += [nn.LayerNorm(self.hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(nn.LayerNorm(self.hidden_layers[i]))
            layers.append(nn.SiLU())

        # Add the final layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class BaseModel:
    def __init__(self,
                 hidden_layers: list,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor) -> None:
        """
        Base model to handle shared logic across different neural network architectures.

        Args:
            hidden_layers (list): List defining the number of neurons in each hidden layer.
            optimizer (str): Optimizer name (e.g., 'adam', 'sgd').
            loss_function (str): Loss function name (e.g., 'MSE', 'MAE').
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and validation.
            train_f (torch.Tensor): Training features.
            train_l (torch.Tensor): Training labels.
        """
        self.input_size = train_f.shape[1]
        self.output_size = train_l.shape[1]
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = NN_Topology(self.input_size, hidden_layers, self.output_size)

        # Define model and optimization
        self.optimizer_str = optimizer
        self.loss_function_str = loss_function

        # Training and validation history
        self.best_model_wts = None
        self.training_loss = []
        self.val_loss = []


    def define_optimizer(self, optimizer):
        if optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters())
        elif optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def fit(self, proc_index, nprocs, path_to_save, model_type, model_ID,
            train_f: Tensor,
            train_l: Tensor,
            val_f: Tensor,
            val_l: Tensor):
        """Train the model using Distributed Data Parallel (DDP)."""
        # Initialize DDP
        dist.init_process_group(backend='nccl', world_size=nprocs, rank=proc_index)
        torch.cuda.set_device(proc_index)

        # Preparing data for DDP
        # Training data
        train_tensor = TensorDataset(train_f, train_l)
        tr_sampler = torch.utils.data.distributed.DistributedSampler(train_tensor,
                                                                     num_replicas=nprocs, rank=proc_index)
        train_loader = torch.utils.data.DataLoader(train_tensor,
                                                   batch_size=self.batch_size, sampler=tr_sampler, num_workers=4)

        # Validation data
        val_tensor = TensorDataset(val_f, val_l)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_tensor,
                                                                      num_replicas=nprocs, rank=proc_index)
        val_loader = torch.utils.data.DataLoader(val_tensor,
                                                 batch_size=self.batch_size, sampler=val_sampler, num_workers=4)

        self.optimizer = self.define_optimizer(self.optimizer_str)
        self.loss_function = define_loss(self.loss_function_str)

        self.model = self.model.to(proc_index)
        model_ddp = DDP(self.model, device_ids=[proc_index], output_device=proc_index)

        best_val_loss = float('inf')
        training_start_time = time.time()
        checkpoint_interval = 1000

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            model_ddp.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(proc_index), labels.to(proc_index)
                self.optimizer.zero_grad()
                outputs = self.forward_with_ddp(model_ddp, inputs)
                loss = self.calculate_loss(outputs, labels, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            avg_training_loss = running_loss / len(train_loader.dataset)
            self.training_loss.append(avg_training_loss)

            # Validation
            val_loss = self.calculate_val_loss(model_ddp, proc_index, val_loader)
            self.val_loss.append(val_loss)
            model_ddp.train()

            # Save the best model weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_wts = model_ddp.state_dict().copy()

            epoch_time = time.time() - epoch_start_time
            if proc_index == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_training_loss:.4e}, "
                      f"Validation Loss: {val_loss:.4e}, Time: {epoch_time:.2f}s")

                # Checkpoint to save model while training
                if epoch % checkpoint_interval == 0 or epoch == self.epochs - 1:
                    instance_path = os.path.join(path_to_save, f'checkpoint_instance_{model_ID}.pk')
                    with open(instance_path, 'wb') as f:
                        pk.dump(self, f)  # Save the entire instance
                    logger.info(f'Checkpoint saved at {instance_path}')

        # Calculate and print the total training time
        total_training_time = time.time() - training_start_time
        if proc_index == 0:
            print(f'Total training time: {total_training_time:.3f}s')
            # Save the model
            self.save_model(path_to_save, model_type, model_ID)

        dist.destroy_process_group()

    @staticmethod
    def forward_with_ddp(model_ddp, inputs):
        """Override if specific forward behavior is required."""
        return model_ddp(inputs)

    def calculate_loss(self, outputs, labels, inputs=None):
        """Override if specific loss behavior is required."""
        return self.loss_function(outputs, labels)

    def calculate_val_loss(self, model_ddp, proc_index, val_loader):
        """Calculate validation loss."""
        model_ddp.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(proc_index), labels.to(proc_index)
                outputs = model_ddp(inputs)
                loss = self.loss_function(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        return val_loss / len(val_loader.dataset)

    def save_model(self, path_to_save, model_type, model_ID, **kwargs):
        """Save the best model weights."""
        if self.best_model_wts is not None:
            # Creates folder to save
            os.makedirs(path_to_save, exist_ok=True)

            # Saves model
            model_path = os.path.join(path_to_save, f"{model_type}{model_ID}.pth")
            torch.save(self.best_model_wts, model_path)
            print(f"Model saved at {model_path}")

            # Saves attributes
            attrs = {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': self.hidden_layers,
                'history': (self.training_loss, self.val_loss),
            }

            # Add additional attributes from kwargs
            for key, value in kwargs.items():
                attrs[key] = value

            attrs_path = os.path.join(path_to_save, f'attrs{model_ID}.pk')
            with open(attrs_path, 'wb') as f:
                pk.dump(attrs, f)

    def predict(self, inputs, proc_index):
        """Run inference."""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size].to(proc_index)
                predictions.append(self.model(batch))
        return torch.cat(predictions, dim=0)

    @staticmethod
    def continue_training(nprocs, path_to_save, model_type, model_ID, epochs,
                          train_f: Tensor,
                          train_l: Tensor,
                          val_f: Tensor,
                          val_l: Tensor):
        instance_path = os.path.join(path_to_save, f"checkpoint_instance_{model_ID}.pk")
        with open(instance_path, 'rb') as f:
            loaded_instance = pk.load(f)

        # Move to appropriate device
        loaded_instance.model = loaded_instance.model.to('cpu')
        loaded_instance.epochs = epochs

        mp.spawn(
            loaded_instance.fit,
            args=(nprocs, path_to_save, model_type, model_ID, train_f, train_l, val_f, val_l),
            nprocs=nprocs,
            join=True)

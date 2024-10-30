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


class PINN_topology(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(PINN_topology, self).__init__()
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
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        x = self.forward(x)
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

    target_moments = torch.zeros((outputs.shape[0], n)) #make sure the outputs.shape[0] is capturing the dimension I want
    target_moments[:, 2] = 1
    target_moments[:, 4] = 1
    physics_loss = (target_moments - moments) ** 2
    return physics_loss.mean(axis=0)


def define_loss(loss_function):
    if loss_function == 'MAE':
        return torch.nn.L1Loss()
    elif loss_function == 'MSE':
        return torch.nn.MSELoss()


class PINN:
    def __init__(self, hidden_layers, optimizer, loss_function, epochs, batch_size, train_f,
                 train_l, val_f, val_l, moments, dynamic_physics_loss=False,
                 alpha_epoch_start=0, alpha_epoch_stop=None, final_alpha=0.5):
        '''alpha_epoch_stop must be bigger than alpha_epoch_start'''
        self.loss_function = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.input_size = train_f.shape[1]
        self.output_size = train_l.shape[1]
        self.hidden_layers = hidden_layers
        self.optim_choice = optimizer
        self.lossfunc_choice = loss_function


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
        self.moments = int(moments)
        self.final_alpha = final_alpha

        # Checking if there will be a dynamic increase of the loss weight
        if dynamic_physics_loss:
            # If yes, self.dynamic_physics_loss is True and the start of the weight is optional, the standard option
            # starts implementing the weight in the first epoch
            self.dynamic_physics_loss = dynamic_physics_loss
            self.alpha_epoch_start = alpha_epoch_start
        else:
            self.dynamic_physics_loss = False

        # If alpha_epoch_stop is given any value, this value will become the final weight increment of the physics loss
        if alpha_epoch_stop is not None:
            self.alpha_epoch_stop = alpha_epoch_stop
        else:
            # If alpha is not given a value, it will be only finished incremented in the last epoch
            self.alpha_epoch_stop = epochs


    def fit(self, proc_index, nprocs, path_to_save, model_type, model_ID):
        logger.debug('Importing backend')
        # Initialising backend
        # If want to use GPU, use nccl backend
        init_process_group(backend='gloo', world_size=nprocs, rank=proc_index)

        # Enumerating process -- Only necessary for GPU
        #torch.cuda.set_device(proc_index)

        logger.debug('Formating data')
        # Preparing data
        train_tensor = TensorDataset(self.train_f, self.train_l)
        tr_sampler = torch.utils.data.distributed.DistributedSampler(train_tensor,
                                                                     num_replicas=nprocs, rank=proc_index)
        self.train_loader = torch.utils.data.DataLoader(train_tensor,
                                                        batch_size=self.batch_size, sampler=tr_sampler, num_workers=4)

        val_tensor = TensorDataset(self.val_f, self.val_l)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_tensor,
                                                                     num_replicas=nprocs, rank=proc_index)
        self.val_loader = torch.utils.data.DataLoader(val_tensor,
                                                      batch_size=self.batch_size, sampler=val_sampler, num_workers=4)

        #self.train_loader = DataLoader(TensorDataset(train_f, train_l), batch_size=batch_size,
        #                               shuffle=True)
        #self.val_loader = DataLoader(TensorDataset(val_f, val_l), batch_size=batch_size,
        #                             shuffle=False)
        logger.debug('Initialising model')
        # Initialising model -- append the end if using GPU
        self.model = PINN_topology(self.input_size, self.output_size, self.hidden_layers)#.to(proc_index)

        # Optimiser
        if self.optim_choice == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())

        # Loss function
        self.loss_function = define_loss(self.lossfunc_choice)

        logger.info('Wrapping model with DDP')
        model_ddp = DDP(self.model)

        # For GPU
        #model_ddp = DDP(self.model, device_ids=[proc_index])

        best_val_loss = float('inf')
        #best_model_wts = None

        training_start_time = time.time()

        training_losses = []
        validation_losses = []

        if self.dynamic_physics_loss:
            alpha = 0
            alpha_increments = (self.final_alpha - alpha) / (self.alpha_epoch_stop - self.alpha_epoch_start)
        else:
            alpha = self.final_alpha

        logger.info('Setting profiler')
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True
        ) as profiler: # whole line for profile

            for epoch in range(self.epochs):
                epoch_start_time = time.time()

                model_ddp.train()  # Set model to training mode
                running_loss = 0.0

                if self.dynamic_physics_loss:
                    if self.alpha_epoch_start <= epoch <= self.alpha_epoch_stop:
                        alpha = alpha + alpha_increments

                for inputs, labels in self.train_loader:
                    self.optimizer.zero_grad()  # Clear previous gradients
                    outputs = model_ddp(inputs)  # Forward pass
                    loss = self.loss_function(outputs, labels)  # Calculate loss

                    physics_loss = physics_loss_fn(outputs, inputs, self.moments)

                    # Below I can implement different weights for the monomial physics loss
                    total_loss = (1 - alpha) * loss + alpha * physics_loss.mean()

                    total_loss.backward()  # Backward pass
                    self.optimizer.step()  # Update labels

                    running_loss += loss.item() * inputs.size(0)

                    profiler.step() # Profile

                avg_training_loss = running_loss / len(self.train_loader.dataset)
                training_losses.append(avg_training_loss)

                # Calculate validation loss after each epoch
                val_loss = calculate_val_loss(model_ddp, self.val_loader, self.loss_function)
                validation_losses.append(val_loss)

                # If want to free up GPU and calc loss on CPU transfer the operation to the CPU
                #val_loss = calculate_val_loss(model_ddp.to('cpu'), self.val_loader, self.loss_function)
                #model_ddp.to(proc_index)

                # Save the model if the validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = model_ddp.state_dict().copy()  # Deep copy the model labels
                    #best_model_wts = {k.replace("module.", ""): v for k, v in model_ddp.state_dict().items()}

                epoch_time = time.time() - epoch_start_time
                if proc_index == 0:
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

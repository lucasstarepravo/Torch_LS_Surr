import torch.optim
from sklearn.model_selection import train_test_split
from data_processing.preprocessing import *
from data_processing.postprocessing import *
from Plots import *
from models.ANN import ANN
from models.PINN import PINN
from models.GP import GP
from models.PINN_ResNet import PINN_ResNet
from models.ResNet import ResNet
from models.SaveNLoad import *
import numpy as np
import pickle as pk


def import_stored_data(file, order, noise):
    amat = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_' + str(order) + '/Noise_' + str(
        noise) + '/Data/amat/amat_' \
           + str(file) + '.csv'
    psi = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_' + str(order) + '/Noise_' + str(
        noise) + '/Data/psi/laplace/psi_' \
          + str(file) + '.csv'
    h = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_' + str(order) + '/Noise_' + str(
        noise) + '/Data/h/h' \
         + str(file) + '.csv'

    amat = np.genfromtxt(amat, delimiter=',', skip_header=0)
    psi = np.genfromtxt(psi, delimiter=',', skip_header=0)
    h = np.genfromtxt(h, delimiter=',', skip_header=0)
    return amat, psi, h[:-1]  # the coefficient multiplying dx comes from Fortran hovdx variable


# Define file details in a list of tuples (file_number, noise)
file_details = [(6, 0.3)]

# Initialize empty lists to store processed data
amat_list = []
psi_list = []

# derivative will be used to standardize and rescale psi
derivative = 'Laplace'

# polynomial will be used to determine which file to import, also for the physics loss function of the PINN,
# and to calculate the moments of the predicted and actual psi
polynomial = 2

for file_number, noise in file_details:
    # Import data
    amat, psi, h = import_stored_data(file_number, order=polynomial, noise=noise)

    stand_psi = standardize_psi(psi, h, derivative)

    # Append processed data to lists
    amat_list.append(amat)
    psi_list.append(stand_psi)

# Concatenate lists to form final datasets
stand_feature = np.concatenate(amat_list, axis=0)
stand_label = np.concatenate(psi_list, axis=0)

train_f, train_l, test_f, test_l, train_index, test_index = create_train_test(stand_feature, stand_label,
                                                                              tt_split=0.9,
                                                                              seed=1)  # This generates the test data

train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=0.2,
                                                  random_state=1)  # This generates the validation data

N = train_l.shape[1]

# Converting data to PyTorch tensors
train_features = torch.tensor(train_f, dtype=torch.float32)
train_labels = torch.tensor(train_l, dtype=torch.float32)
val_features = torch.tensor(val_f, dtype=torch.float32)
val_labels = torch.tensor(val_l, dtype=torch.float32)

#pinn1 = PINN([64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
#             optimizer='adam',
#             loss_function='MSE', epochs=1000, batch_size=64, train_f=train_features,
#             train_l=train_labels, val_f=val_features, val_l=val_labels, moments=polynomial, final_alpha=0.3)
#pinn1.fit()

#ann = ANN([64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], optimizer='adam', loss_function='MSE',
#          epochs=1500, batch_size=64, train_f=train_features, train_l=train_labels, val_f=val_features,
#          val_l=val_labels)
hidden_layers = 100 * [32]

res2 = ResNet(hidden_layers, optimizer='adam',
             loss_function='MSE', epochs=2500, batch_size=128, train_f=train_features, train_l=train_labels,
             val_f=val_features, val_l=val_labels, skip_connections=[(0, 99), (1, 19), (20, 39), (40, 59), (60, 79), (80, 98)])

pinn_res2 = PINN_ResNet(hidden_layers,
                       optimizer='adam', loss_function='sgs', epochs=3000, batch_size=128, train_f=train_features,
                       train_l=train_labels, val_f=val_features, val_l=val_labels,
                       skip_connections=[(0, 99), (1, 19), (20, 39), (40, 59), (60, 79), (80, 98)], moments=polynomial,
                       final_alpha=0.5, alpha_epoch_start=1000)

ann.fit()

# gp = GP(train_x=train_features, train_y=train_labels)

# To save a model
# Here the file path should be the directory which the history and model labels should be saved
# save_model_instance(ann, 'file path', 'ann or pinn', 'model_ID')


# To load a model
# Here the file path should be the exact path to the file and should contain the file in the end of the directory
# Notice that the file path to model and attrs will contain different files in the end
# attrs = load_attrs('file_path') # Recall to add attrs#.pk at the end of the path
# model = load_model_instance('file_path', attrs)


plot_training_pytorch(ann)

test_features = torch.tensor(test_f, dtype=torch.float32)
pred_l = ann.predict(test_features)

# scaled_psi_pred, scaled_psi_act = rescale_psi(pred_l, test_l, h, derivative) # Right now h can only rescale 1 resolution at a time
moments_act = calc_moments(test_f, test_l, polynomial=polynomial)
moments_pred = calc_moments(test_f, pred_l.detach().numpy(), polynomial=polynomial)
moment_error = np.mean(abs(moments_pred - moments_act), axis=0)
moment_std = np.std(abs(moments_pred - moments_act), axis=0)

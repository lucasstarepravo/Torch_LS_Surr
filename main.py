import torch.optim
from sklearn.model_selection import train_test_split
from learning_shapefunction.data_processing.preprocessing import *
from learning_shapefunction.data_processing.postprocessing import *
from learning_shapefunction.Plots import *
from learning_shapefunction.models.ANN import ANN
from learning_shapefunction.models.PINN import PINN
from learning_shapefunction.models.GP import GP
from learning_shapefunction.models.SaveNLoad import *
import numpy as np
import pickle as pk


def import_stored_data(file, order, noise):
    ij_link_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/neigh/ij_link' \
                   + str(file) + '.csv'
    coor_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/coor/coor' \
                + str(file) + '.csv'
    weights_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/weights/laplace/w_'\
                   + str(file) + '.csv'
    dx_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/dx/dx' \
              + str(file) + '.csv'

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]
    weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)
    weights = trim_zero_columns(weights[:, 1:])
    dx = np.genfromtxt(dx_path, delimiter=',', skip_header=0)
    dx = dx[0]
    return ij_link, coor, weights, dx


# Define file details in a list of tuples (file_number, noise)
file_details = [(7, 0.5), (7, 0.4)]

# Initialize empty lists to store processed data
features_list = []
weights_list = []
coor_list = []

for file_number, noise in file_details:
    # Import data
    ij_link, coor, weights, dx = import_stored_data(file_number, order=2, noise=noise)

    # Extract and process features
    features = feat_extract(coor, ij_link)
    features = features[:, 1:, :]  # Removes the first item which is always 0

    # Append processed data to lists
    features_list.append(features)
    weights_list.append(weights)
    coor_list.append(coor)

# Concatenate lists to form final datasets
features = np.concatenate(features_list, axis=0)
weights = np.concatenate(weights_list, axis=0)
coor = np.concatenate(coor_list, axis=0)


stand = 5 # stand determines the type of normalization that will be applied
if stand == 1:
    stand_feature, stand_label, f_mean, f_stdv, l_mean, l_stdv = standardize_comp_stencil(features,
                                                                                          weights)
elif stand == 2:
    stand_feature, stand_label, f_mean, l_mean, h_scale_xy, h_scale_w = non_dimension(features, weights,
                                                                                      dx, dtype='laplace')
elif stand == 3:
    stand_feature, stand_label, f_mean, f_stdv, l_mean, l_stdv = global_standard(features,
                                                                                 weights)
elif stand == 4:
    stand_feature, stand_label, f_mean, f_stdv, l_stdv = std_dev_norm(features, weights,
                                                                      'laplace')
elif stand == 5:
    stand_feature, stand_label,f_stdv, l_stdv = spread_norm(features, weights,
                                                            'laplace')

polynomial = 2
monomial_stand_feature = monomial_expansion(stand_feature, polynomial=polynomial)


train_f, train_l, test_f, test_l, train_index, test_index = create_train_test(monomial_stand_feature, stand_label,
                                                                              tt_split=0.9, seed=1) # This generates the test data

train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=0.2, random_state=1) # This generates the validation data


N = train_l.shape[1]

# Converting data to PyTorch tensors
train_features = torch.tensor(train_f, dtype=torch.float32)
train_labels = torch.tensor(train_l, dtype=torch.float32)
val_features = torch.tensor(val_f, dtype=torch.float32)
val_labels = torch.tensor(val_l, dtype=torch.float32)

ann = PINN([256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512], optimizer='adam',
            loss_function='MSE', epochs=1000, batch_size=64, train_f=train_features,
            train_l=train_labels, val_f=val_features, val_l=val_labels, moments=2, dynamic_physics_loss=True,
            alpha_epoch_start=200, alpha_epoch_stop=600, final_alpha=0.4)

#ann = ANN([256, 256, 256, 256, 256, 256, 256], optimizer='adam', loss_function='MSE',
#          epochs=600, batch_size=64, train_f=train_features, train_l=train_labels, val_f=val_features, val_l=val_labels)

#gp = GP(train_x=train_features, train_y=train_labels)

ann.fit()

# To save a model
# Here the file path should be the directory which the history and model weights should be saved
# save_model_instance(ann, 'file path', 'ann or pinn', 'model_ID')


# To load a model
# Here the file path should be the exact path to the file and should contain the file in the end of the directory
# Notice that the file path to model and attrs will contain different files in the end
# attrs = load_attrs('file_path')
# model = load_model_instance('file_path', attrs)


plot_training_pytorch(ann)

test_features = torch.tensor(test_f, dtype=torch.float32)
pred_l = ann.predict(test_features)

'''The two functions below rescale the data to their original magnitude'''
if stand == 1:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_stencil(test_l, pred_l, test_f, f_mean, f_stdv, l_mean,
                                                                  l_stdv, test_index)
elif stand == 2:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_h(test_l, pred_l, test_f, f_mean, l_mean, h_scale_xy,
                                                        h_scale_w, test_index)
elif stand == 3:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_global_stand(test_l, pred_l, test_f, f_mean, f_stdv, l_mean,
                                                                       l_stdv, test_index)
elif stand == 4:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_std(test_l, pred_l, test_f, f_mean, f_stdv,
                                                              l_stdv, test_index)
elif stand == 5:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_spread(test_l, pred_l, test_f,
                                                                 f_stdv, l_stdv, test_index, polynomial)


test_neigh_coor = d_2_c(coor, test_index, scaled_feat)

plot_node_prediction_error(scaled_pred_l, scaled_actual_l, test_neigh_coor, node='random', size=80, option=3)


pred = error_test_func(scaled_feat, scaled_pred_l)
act = error_test_func(scaled_feat, scaled_actual_l)
err = act - pred
err_mean = np.mean(err)
err_std = np.std(err)

point_v_actual = calc_moments(scaled_feat, scaled_actual_l, polynomial=2)
point_v_pred = calc_moments(scaled_feat, scaled_pred_l.numpy(), polynomial=2)
moment_error = np.mean(abs(point_v_pred - point_v_actual), axis=0)
moment_std = np.std(abs(point_v_pred - point_v_actual), axis=0)

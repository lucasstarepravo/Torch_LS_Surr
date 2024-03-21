import numpy as np
import matplotlib.pyplot as plt


def plot_training_pytorch(history):
    try:
        training_losses = history.training_loss
        validation_losses = history.val_loss
    except AttributeError:
        training_losses, validation_losses = history['history']

    # Create a range of epochs for the x-axis
    epochs = range(1, len(training_losses) + 1)

    # Plot the training and validation loss
    plt.plot(epochs, training_losses, 'b', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# This function will be used to plot 1 or 3 graphs, either only one graph with the % prediction error, or the one just
# mentioned + 2 graphs (i) showing the actual weight and the node positions and the other (ii) showing the predicted
# weights and positions
def plot_node_prediction_error(pred_l, actual_l, coor_subset, node='random', size=20, option=1):
    '''features is supposed to be the test set of features already scaled back and with coordinates, NOT distance'''
    '''option variable controls whether to return only 1 graph with error of weight prediction or the 3 graphs
    1 - returns only 1 graph with the absolute error of prediction
    2 - return only 1 graph with the % error of prediction
    3 - returns 3 graphs (i) predicted weights (ii) actual weights (iii) absolute error weights'''

    N = len(pred_l)
    if node == 'random':
        plot_i = np.random.randint(0, N)
    else:
        plot_i = int(N)

    features_node = coor_subset[plot_i, :]
    pred_l_node = pred_l[plot_i, :]
    actual_l_node = actual_l[plot_i, :]
    error = pred_l_node - actual_l_node

    features_node = features_node.reshape(-1, 2)
    ref_node = features_node[0, :]
    neigh_nodes = features_node[1:, :]

    if option == 1:
        plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size)
        plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=abs(error), label='Neighbour Nodes', s=size)
        plt.colorbar()
        plt.legend()
        plt.show()
        return
    elif option == 2:
        #This option plots percentage error, which is not ideal
        error = (pred_l_node - actual_l_node)/actual_l_node
        plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size)
        plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=abs(error), label='Neighbour Nodes', s=size)
        plt.title('Weight Percentage Error')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()
        plt.legend()
        plt.show()
    else:
        plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size)
        plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=abs(error), label='Neighbour Nodes', s=size)
        plt.title('Weight Error')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()
        plt.legend()
        plt.show()

        plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size)
        plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=pred_l_node, label='Predicted W', s=size)
        plt.title('Predicted Weight')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()
        plt.legend()
        plt.show()

        plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size)
        plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=actual_l_node, label='Actual W', s=size)
        plt.title('Actual Weight')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()
        plt.legend()
        plt.show()
        return


def plot_c(x_axis, y_axis, optimal_c):
    # Calculate y values for the best fit line using the given optimal_c
    x_range = np.linspace(min(x_axis), max(x_axis), 100)
    y_best_fit = (1 / (optimal_c * x_range))**2

    # Plotting
    plt.figure(figsize=(10, 6))
    # Use 'x' marker for more precise indication of points
    plt.plot(x_range, y_best_fit, 'r-', label=f'Best Fit Line (c={optimal_c:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Initial Points with Optimal c')
    plt.legend()
    plt.grid(True)
    plt.show()

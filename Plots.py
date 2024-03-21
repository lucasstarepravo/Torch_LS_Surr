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

    # Set y-axis to log scale
    plt.yscale('log')

    plt.ylabel('Loss')
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_training_pytorch(history, window_size=10, plot_original=False, save=False, show_legend=True, log_y=False,
                          title=False, plot_smooth=False):
    plt.style.use('seaborn-v0_8-paper')

    try:
        training_losses = history.tr_loss
        validation_losses = history.val_loss
    except AttributeError:
        try:
            training_losses = history['tr_loss']
            validation_losses = history['val_loss']
        except KeyError:
            training_losses, validation_losses = history['history']

    # Smooth the validation losses using a moving average
    def moving_average(data, size):
        # Ensure the moving average does not go beyond the data length
        if size < 1:
            size = 1
        return np.convolve(data, np.ones(size) / size, mode='same')

    smoothed_validation_losses = moving_average(validation_losses, window_size)

    # Create a range of epochs for the x-axis
    epochs = range(1, len(training_losses) + 1)

    # Create the plot with high quality settings
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the training and original validation loss
    ax.plot(epochs, training_losses, linestyle='-', color='b', label='Training Loss')
    if plot_original:
        ax.plot(epochs, validation_losses, linestyle='-', color='r', alpha=0.5, label='Validation Loss (Original)')

    # Plot the smoothed validation loss
    if plot_smooth:
        ax.plot(epochs, smoothed_validation_losses, linestyle='-', color='g', alpha=0.5, label='Validation Loss (Smoothed)', linewidth=2)

    if title==False:
        plt.title('Training and Validation Loss', fontsize=16, family='serif')
    else:
        plt.title(title, fontsize=16, family='serif')
    plt.xlabel('Epochs', fontsize=16, family='serif')
    plt.ylabel('Loss (MSE)', fontsize=16, family='serif')

    # Set y-axis to logarithmic scale if specified
    if log_y:
        ax.set_yscale('log')

    # Set axis tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    # Add the legend
    if show_legend:
        legend = ax.legend(loc='best', fontsize=12, prop={'family': 'serif'})
        plt.setp(legend.get_texts(), fontsize='12', family='serif')

    # Display the grid
    ax.grid(True, which="both", linestyle='-', color='gray', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.5, alpha=0.3)
    ax.minorticks_on()

    plt.tight_layout()

    if save:
        # Save the figure as a PDF with high resolution
        plt.savefig('training_plot_pytorch_high_quality.pdf', format='pdf', dpi=300)

    plt.show()
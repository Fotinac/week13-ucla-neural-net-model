import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot training loss and accuracy curves from Keras model training history.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    axs[0].plot(history.history['accuracy'], label='Accuracy')
    axs[0].set_title('Training Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot loss
    axs[1].plot(history.history['loss'], label='Loss', color='orange')
    axs[1].set_title('Training Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    fig.tight_layout()
    return fig

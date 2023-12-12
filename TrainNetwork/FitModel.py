import matplotlib.pyplot as plt
import os
from TrainNetwork.BuildResultsDir import build_results_dir

def fit_model(model, X_train, y_train, epochs=250, batch_size=16, validation_split=0.2, model_name='none'):
    """
    Train the model using the given training data and plot training/validation metrics.

    Parameters:
    model (tf.keras.Model): The model to be trained.
    X_train (array-like): Training data features.
    y_train (array-like): True labels for the training data.
    epochs (int): Number of epochs for training. Default is 250.
    batch_size (int): Size of mini-batches. Default is 16.
    validation_split (float): Fraction of training data to be used for validation. Default is 0.2.
    model_name (str): Name of the model. Default is 'none'.

    Returns:
    tf.keras.Model: Trained model.
    """
    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split)

    # Get metrics names
    train_metrics = model.metrics_names

    # Plot metrics
    plt.figure(figsize=(15, 10), dpi=250)
    plt.rcParams['axes.facecolor'] = 'lightgray'  # Set background color

    for i, metric in enumerate(train_metrics):
        plt.subplot(1, len(train_metrics), i + 1)

        # Plot training and validation metrics
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])

        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.grid(which='major', color='#dddddd')
        plt.grid(which='minor', color='#eeeeee')
        plt.minorticks_on()

    # Create results directory
    results_dir = build_results_dir(model_name)

    # Save plot
    plot_path = os.path.join(results_dir, 'Network_loss' + '_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return model

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def create_MNIST_model(input_dim, num_categories, name=None):
    """
    Create a 2-layer Sequential model for MNIST digit-detect prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 256, 128 neurons respectively with a ReLU activation function
    # The final layer has a `num_categories` neuron with a softmax activation function
    model.add(Dense(256, activation="relu", input_dim=input_dim, name="Hidden1"))
    model.add(Dense(128, activation="relu", name="Hidden2"))
    model.add(Dense(num_categories, activation="softmax", name="output"))

    # Print model info on console
    model.summary()

    # Compile the model with categorical_crossentropy loss function, rmsprop optimizer, and categorical_accuracy metrics
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["categorical_accuracy"],
    )

    return model


def train_evaluate_save_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    num_categories,
    batch_size=32,
    name="model",
    epochs=20,
):
    """
    Train, evaluate, and save the MNIST digit-detect prediction model.

    Args:
        X_train (pandas.DataFrame): Input features for training.
        y_train (pandas.Series): Output values for training.
        X_valid (pandas.DataFrame): Input features for validation.
        y_valid (pandas.Series): Output values for validation.
        X_test (pandas.DataFrame): Input features for testing.
        y_test (pandas.Series): Output values for testing.
        batch_size (int): Batch size for training and evaluation.
        name (str): Name for saving the model.
        epochs (int): Number of training epochs.

    Returns:
        float: categorical accuracy of the model on the test dataset.
        numpy.ndarray: Predicted MNIST digit-detect.
    """

    # Train the model using the custom data generator for training, test, and validation data
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)

    # Evaluate the model on the test dataset
    loss, categorical_accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save(f"{name}.h5")

    return hist, loss, categorical_accuracy


def plot_metric_graph(hist, metric, title):
    """
    Plot the training and validation metric as a function of epochs.

    Args:
        hist (keras.callbacks.History): Training history object.
        metric (str): Name of the metric to plot.
        title (str): Title for the plot.
    """
    # Create plot with custom styling
    plt.figure(figsize=(15, 5))
    plt.plot(
        hist.epoch,
        hist.history[metric],
        linewidth=2,
        color="blue",
        label=f"training {metric}",
    )
    plt.plot(
        hist.epoch,
        hist.history[f"val_{metric}"],
        linewidth=2,
        color="orange",
        label=f"validation {metric}",
    )
    plt.title(f"Epochs vs {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.grid(alpha=0.5)
    plt.legend()

    # Save the plot as a JPEG file
    plt.savefig(f"{title}.jpg", dpi=300, bbox_inches="tight")

    # Show plot
    plt.show()


def get_prediction_images(model, folder_name, ext):
    for path in glob.glob(f"{folder_name}/*.{ext}"):
        img_data = plt.imread(path)
        grayed = np.average(img_data, axis=2, weights=[0.3, 0.59, 0.11]) / 255
        # Reshape and replicate across three channels to match the expected input shape
        grayed_replicated = np.repeat(grayed.reshape(32, 32, 1), 3, axis=2)
        prediction = model.predict(grayed_replicated.reshape(1, 32, 32, 3))
        print(f"{path}: {np.argmax(prediction)}")


def main():
    """
    Main function to train and evaluate the MNIST prediction model.
    """
    # Read MNIST train and test data from data file as pandas DataFrame
    MNIST_train_data = pd.read_csv("mnist_train.csv")
    MNIST_test_data = pd.read_csv("mnist_test.csv")

    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = MNIST_train_data.iloc[:, 1:].to_numpy()
    y_train = MNIST_train_data.iloc[:, 0].to_numpy()

    # Separate the input features (X_test) and output values (y_test) of the test dataset
    X_test = MNIST_test_data.iloc[:, 1:].to_numpy()
    y_test = MNIST_test_data.iloc[:, 0].to_numpy()

    # Get labels
    labels = np.unique(y_test)

    # Scale X_train and X_test
    X_train = X_train / 255
    X_test = X_test / 255

    # Apply one hot encoding to y_train and y_test
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Get the number of output neurons
    num_categories = len(labels)

    # Create model using create_MNIST_model func.
    model = create_MNIST_model(
        input_dim=X_train.shape[1],
        num_categories=num_categories,
        name="MNIST-digit-detect",
    )

    # Train and evaluate the machine learning model using the training and test data
    hist, loss, categorical_accuracy = train_evaluate_save_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_categories,
        name="MNIST-digit-detect",
    )

    # Free up memory
    del (
        MNIST_train_data,
        MNIST_test_data,
        X_train,
        X_test,
        y_train,
        y_test,
        labels,
        num_categories,
    )

    # Plot the loss and categoracal accuracy for each epoch during training
    plot_metric_graph(hist, metric="loss", title="Epoch-Loss Graph")
    plot_metric_graph(
        hist, metric="categorical_accuracy", title="Epoch-categorical Accuracy Graph"
    )

    # Print the test loss and categorical_accuracy of the trained model
    print("################################")
    print("Model Evaluation Metrics:")
    print(f"loss: {loss:.2f}\ncategorical_accuracy: {categorical_accuracy:.2f}")
    print("################################")

    # Print the predicted detection for each individual in the folder
    get_prediction_images(model, "predicted", "jpg")


if __name__ == "__main__":
    main()

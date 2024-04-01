"""
This module provides functionality to load and concatenate CIFAR-10 dataset files into 
a single NumPy array. It includes functions to unpickle CIFAR-10 data files and concatenate 
them, handling both training and test datasets. Additionally, it defines a function to create 
a convolutional neural network model for CIFAR-10 image classification.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def unpickle(file_path: str) -> Dict[bytes, Any]:
    """
    Unpickle a CIFAR dataset file.

    Args:
        file_path (str): The path to a CIFAR dataset file.

    Returns:
        Dict[bytes, Any]: The contents of the unpickled file.
    """

    with open(file_path, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def concat_datasets(dataset_paths: List[str]) -> np.ndarray:
    """
    Concatenates datasets from a list of CIFAR dataset file paths into a single NumPy array,
    combining both data and labels.

    Parameters:
        dataset_paths (List[str]): A list of paths to the CIFAR dataset files.

    Returns:
        np.ndarray: A NumPy array containing concatenated data and labels, where data has
        shape (N, 3072) and labels are appended as the last column, resulting in shape
        (N, 3073) for the array.
    """

    data_arrays = []
    label_arrays = []

    for path in dataset_paths:
        dataset = unpickle(path)
        data_arrays.append(np.array(dataset[b"data"], dtype=np.float32))
        label_arrays.append(np.array(dataset[b"labels"], dtype=np.float32))

    combined_data = np.vstack(data_arrays)
    combined_labels = np.hstack(label_arrays)

    # Note that combined_labels need to be reshaped to be concatenated along the second axis
    combined_dataset = np.hstack((combined_data, combined_labels.reshape(-1, 1)))

    return combined_dataset


def create_cifar_model(
    filter_shape: tuple, input_shape: tuple, name: str = "CIFAR_Model"
) -> Model:
    """
    Creates a Sequential model for CIFAR-10 image classification.

    Parameters:
        filter_shape (tuple): The dimension of the convolution filters.
        input_shape (tuple): The shape of the input images.
        name (str, optional): The name of the model.

    Returns:
        Model: A compiled Keras model ready for training.
    """
    model = Sequential(name=name)

    model.add(
        Conv2D(
            32,
            filter_shape,
            activation="relu",
            input_shape=input_shape,
            name="Conv2D-1",
        )
    )
    model.add(MaxPooling2D(name="MaxPooling-1"))
    model.add(Conv2D(64, filter_shape, activation="relu", name="Conv2D-2"))
    model.add(MaxPooling2D(name="MaxPooling-2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(256, activation="relu", name="Dense-1"))
    model.add(Dense(128, activation="relu", name="Dense-2"))
    model.add(Dense(10, activation="softmax", name="Output"))

    # Print model info on console
    model.summary()

    # Compile the model with categorical_crossentropy loss function, rmsprop optimizer, and categorical_accuracy metrics
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["categorical_accuracy"],
    )

    return model


def train_evaluate_save_cifar_model(
    model: Model,
    X_train: np.ndarray,
    y_train_one_hot: np.ndarray,
    X_test: np.ndarray,
    y_test_one_hot: np.ndarray,
    batch_size: int = 32,
    name: str = "cifar_model",
    epochs: int = 20,
) -> Tuple[Dict[str, list], float, float]:
    """
    Train, evaluate, and save the CIFAR-10 image classification model.

    Args:
        model (Model): The Keras model to be trained and evaluated.
        X_train (np.ndarray): Input features for training, normalized.
        y_train_one_hot (np.ndarray): One-hot encoded output values for training.
        X_test (np.ndarray): Input features for testing, normalized.
        y_test_one_hot (np.ndarray): One-hot encoded output values for testing.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        name (str, optional): Name for saving the model. Defaults to 'cifar_model'.
        epochs (int, optional): Number of training epochs. Defaults to 20.

    Returns:
         Tuple[Dict[str, list], float, float]:  Training history object containing recorded metrics
                                                for each epoch, loss of the model on the test
                                                dataset, and categorical accuracy of the model on
                                                the test dataset.
    """

    # Train the model
    hist = model.fit(
        X_train,
        y_train_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )

    # Evaluate the model on the test dataset
    loss, categorical_accuracy = model.evaluate(
        X_test, y_test_one_hot, batch_size=batch_size, verbose=0
    )

    # Save the model
    model.save(f"{name}.h5")

    # Optionally, you might want to return the entire history object and the model's performance metrics
    return hist.history, loss, categorical_accuracy


def main():
    # Define the directory containing CIFAR-10 batch files
    cifar_10_batches_dir = os.path.join(
        os.getcwd(), "12-CIFAR10-Classification", "cifar-10-batches-py"
    )

    # Build the training dataset paths
    batch_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    X_dataset_paths = [
        os.path.join(cifar_10_batches_dir, batch_file) for batch_file in batch_files
    ]

    # Build the test dataset path(s)
    y_dataset_path = [os.path.join(cifar_10_batches_dir, "test_batch")]

    # Load and concatenate training and test datasets
    train_dataset = concat_datasets(X_dataset_paths)
    test_dataset = concat_datasets(y_dataset_path)

    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]

    # Separate the input features (X_test) and output values (y_test) of the test dataset
    X_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

    # Correct Reshaping to (32, 32, 3)
    X_train_reshaped = X_train.reshape(-1, 32, 32, 3)
    X_test_reshaped = X_test.reshape(-1, 32, 32, 3)

    # Normalize pixel values
    X_train_normalized = X_train_reshaped / 255.0
    X_test_normalized = X_test_reshaped / 255.0

    # One-hot encode labels
    y_train_one_hot = to_categorical(y_train, 10)
    y_test_one_hot = to_categorical(y_test, 10)

    # Create CIFAR model instance
    cifar_model = create_cifar_model(filter_shape=(3, 3), input_shape=(32, 32, 3))


if __name__ == "__main__":
    main()

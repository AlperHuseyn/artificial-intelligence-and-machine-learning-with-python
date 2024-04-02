import glob
import os
import pickle
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def unpickle(
    file_path: str, meta_data: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[List[bytes], List[bytes]]]:
    """
    Unpickle a CIFAR dataset or metadata file.

    Args:
        file_path (str): The path to a CIFAR dataset or metadata file.
        meta_data (bool, optional): Flag indicating whether the file is a metadata file. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[List[bytes], List[bytes]]]:
        If meta_data is False, returns a tuple containing three numpy arrays:
            - Image data as numpy array with shape (number_of_samples, 3072).
            - Coarse labels as numpy array with shape (number_of_samples,).
            - Fine labels as numpy array with shape (number_of_samples,).
        If meta_data is True, returns a tuple containing two lists of bytes:
            - Coarse label names.
            - Fine label names.
    """

    with open(file_path, "rb") as fo:
        dataset = pickle.load(fo, encoding="bytes")

    if meta_data:
        coarse_label_names = dataset[b"coarse_label_names"]
        fine_label_names = dataset[b"fine_label_names"]

        # Decode bytes to strings for readability
        coarse_label_names = [label.decode("utf-8") for label in coarse_label_names]
        fine_label_names = [label.decode("utf-8") for label in fine_label_names]

        return coarse_label_names, fine_label_names

    # Each image in CIFAR-100 is represented as a 3072-dimensional vector
    # Coarse and fine labels are both 1-dimensional vectors
    img_data = np.array(dataset[b"data"], dtype=np.float32)
    coarse_lables = np.array(dataset[b"coarse_labels"], dtype=np.int32)
    fine_lables = np.array(dataset[b"fine_labels"], dtype=np.int32)

    return img_data, coarse_lables, fine_lables


def create_cifar_model(
    filter_shape: tuple, input_shape: tuple, name: str = "CIFAR_Model"
) -> Model:
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
    model.add(Conv2D(128, filter_shape, activation="relu", name="Conv2D-3"))
    model.add(MaxPooling2D(name="MaxPooling-3"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(512, activation="relu", name="Dense-1"))
    model.add(Dense(256, activation="relu", name="Dense-2"))
    model.add(Dense(100, activation="softmax", name="Output"))

    return model


def compile_and_summarize_model(
    model: Model,
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
) -> None:
    """
    Compiles the Keras model with the given parameters and prints its summary.

    Args:
        model (Model): The Keras model to be compiled.
        optimizer (str, optional): The name of the optimizer to use. Defaults to 'rmsprop'.
        loss (str, optional): The loss function to be used. Defaults to 'categorical_crossentropy'.
        metrics (list, optional): The list of metrics to be evaluated by the model during training
        and testing. Defaults to ['categorical_accuracy'].

    Returns:
        None
    """

    # Compile the model with the specified parameters
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Print the model summary to review its architecture
    model.summary()


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

    # Optionally, you might want to return the entire history object and
    # the model's performance metrics
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


def get_prediction_images(model, folder_name, ext, class_names):
    for path in glob.glob(f"{folder_name}/*.{ext}"):
        # Load image using Keras, resize it to 32x32 pixels (the input shape the model expects)
        # and convert it to an array.
        img = load_img(path, target_size=(32, 32))
        img_array = img_to_array(img)

        # The model expects a 4D batch as input, so we add a dimension with np.expand_dims
        img_batch = np.expand_dims(img_array, axis=0)

        # Normalize the image data to 0-1
        img_batch /= 255.0

        # Make predictions
        prediction = model.predict(img_batch)
        predicted_class = class_names[np.argmax(prediction)]
        print(f"{path}: {predicted_class}")


def main():
    # Define the directory containing CIFAR-100 batch files
    cifar_100_batches_dir = os.path.join(
        os.getcwd(), "13-Checkpoint-EarlyStop-Techniques", "cifar-100-python"
    )

    # Build the training dataset path
    X_dataset_path = os.path.join(cifar_100_batches_dir, "train")
    # Build the test dataset path
    y_dataset_path = os.path.join(cifar_100_batches_dir, "test")

    # Load training and test datasets
    X_train, _, y_train = unpickle(X_dataset_path)
    X_test, _, y_test = unpickle(y_dataset_path)

    # Correct Reshaping to (32, 32, 3)
    X_train_reshaped = X_train.reshape(-1, 32, 32, 3)
    X_test_reshaped = X_test.reshape(-1, 32, 32, 3)

    # Normalize pixel values
    X_train_normalized = X_train_reshaped / 255.0
    X_test_normalized = X_test_reshaped / 255.0

    # One-hot encode labels
    y_train_one_hot = to_categorical(y_train, 100)
    y_test_one_hot = to_categorical(y_test, 100)

    # Create CIFAR model instance
    cifar_model = create_cifar_model(filter_shape=(3, 3), input_shape=(32, 32, 3))

    # Compile and summarize the model
    compile_and_summarize_model(
        cifar_model,
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # Train and evaluate the machine learning model using the training and test data
    hist, loss, categorical_accuracy = train_evaluate_save_cifar_model(
        cifar_model,
        X_train_normalized,
        y_train_one_hot,
        X_test_normalized,
        y_test_one_hot,
        batch_size=32,
        name="cifar_model",
        epochs=5,
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

    # Load the label names from batches.meta
    _, fine_label_names = unpickle(
        os.path.join(cifar_100_batches_dir, "meta"), meta_data=True
    )

    # Print the predicted detection for each individual in the folder
    get_prediction_images(cifar_model, "test-images", "jpg", fine_label_names)


if __name__ == "__main__":
    main()

"""
This script trains and evaluates a machine learning model for sentiment prediction using the IMDB dataset. The main focus of this script is to demonstrate the process of vectorizing text data and building a basic neural network model. It does not aim to achieve high performance or accuracy. However, by improving the vectorization techniques, such as using more advanced text preprocessing methods or employing pre-trained word embeddings, and optimizing the model architecture and hyperparameters, you can aim for better performance and accuracy in sentiment prediction tasks.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Define constants
MAX_VOCAB_SIZE: int = 10000  # Top most frequent words to consider
MAX_LENGTH: int = 300  # Maximum length of all sequences
EMBEDDING_DIM: int = 64  # Dimension of the word embeddings


def create_IMDB_model(input_dim: int = MAX_VOCAB_SIZE, name=None) -> Sequential:
    """
    Create a Sequential model for IMDB review sentiment prediction with an Embedding layer.

    Args:
    - input_dim: int, dimension of the input vector.
    - name: str, name of the model.

    Returns:
    - Sequential Keras model.
    """
    model = Sequential(name=name)
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH,
            name="Embedding",
        )
    )
    model.add(Conv1D(64, 3, padding="same", activation="relu", name="Conv1D-1"))
    model.add(MaxPooling1D(2, name="MaxPooling1D-1"))
    model.add(Conv1D(128, 3, padding="same", activation="relu", name="Conv1D-2"))
    model.add(MaxPooling1D(2, name="MaxPooling1D-2"))
    model.add(Flatten(name="Flatten"))  # Flatten the output of the Embedding layer
    # Define the architecture of the neural network by adding layers to the model
    # The first two layers each with a ReLU activation function
    # The final layer with a sigmoid activation function
    model.add(Dense(256, activation="relu", name="Hidden1"))
    model.add(Dense(128, activation="relu", name="Hidden2"))
    model.add(Dense(1, activation="sigmoid", name="output"))

    # Print model info on console
    model.summary()

    # Compile the model with binary_crossentropy loss function, rmsprop optimizer, and binary_accuracy metrics
    model.compile(
        loss="binary_crossentropy", optimizer="rmsprop", metrics=["binary_accuracy"]
    )

    return model


def train_evaluate_save_model_(
    model: Model,
    X_train: np.ndarray,
    y_train_one_hot: np.ndarray,
    X_test: np.ndarray,
    y_test_one_hot: np.ndarray,
    X_to_predict,
    batch_size: int = 32,
    name: str = "model",
    epochs: int = 20,
    patience: float = 5.0,
    min_delta: float = 0.0,
) -> Tuple[Dict[str, list], float, float]:
    """
    Train, evaluate, and save the model with early stopping and checkpoint.

    Args:
    - model: Keras Model, the model to be trained and evaluated.
    - X_train, y_train: training data and labels.
    - X_test, y_test: test data and labels.
    - X_to_predict: data to be predicted.
    - batch_size: int, batch size for training.
    - name: str, name for saving the model.
    - epochs: int, number of training epochs.
    - patience: int, early stopping patience.
    - min_delta: float, early stopping min delta.

    Returns:
    - Training history, test loss, and test accuracy.
    """

    # Initialize EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=patience,  # How many epochs to wait after last time val_loss improved
        verbose=1,
        min_delta=min_delta,  # Minimum change to qualify as an improvement
        mode="min",  # The training will stop when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity.
    )

    model_checkpoint = ModelCheckpoint(
        os.path.join("model_files", "checkpoint.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    # Train the model
    hist = model.fit(
        X_train,
        y_train_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Evaluate the model on the test dataset
    loss, binary_accuracy = model.evaluate(
        X_test, y_test_one_hot, batch_size=batch_size, verbose=0
    )

    # Predict review-sentiment from comments
    predictions = model.predict(X_to_predict)

    # Save the model
    model.save(os.path.join("model_files", f"{name}.keras"))

    # Optionally, you might want to return the entire history object and
    # the model's performance metrics
    return hist, loss, binary_accuracy, predictions


def plot_metric_graph(hist, metric: str, title: str):
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

    plot_dir = "model_files"
    os.makedirs("model_files", exist_ok=True)

    # Save the plot as a JPEG file
    plt.savefig(os.path.join(plot_dir, f"{title}.jpg"))

    # Show plot
    plt.show()


def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    # Read imdb data from data file as pandas DataFrame
    IMDB_data = pd.read_csv("IMDB Dataset.csv")

    # Encode sentiment column (0 - negative, 1 - positive)
    encoder = LabelEncoder()
    y_dataset = encoder.fit_transform(IMDB_data["sentiment"]).astype(np.int8)

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(IMDB_data["review"])
    sequences = tokenizer.texts_to_sequences(IMDB_data["review"])

    # Pad the sequences
    X_dataset = pad_sequences(sequences, maxlen=MAX_LENGTH)

    # Split the dataset into training and test sets using train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)

    # Load the array to be predicted and perform feature scaling on it
    IMDB_data_to_predict = pd.read_csv("predicted.csv")

    # Apply tokenization to the data array for prediction
    IMDB_data_to_predict_sequences = tokenizer.texts_to_sequences(
        IMDB_data_to_predict["review"]
    )
    IMDB_data_to_predict_vector = pad_sequences(
        IMDB_data_to_predict_sequences, maxlen=MAX_LENGTH
    )

    # Create the IMDB model and compile it
    model = create_IMDB_model()

    # Train and evaluate the machine learning model using the training and test data
    hist, loss, binary_accuracy, predictions = (
        train_evaluate_save_model_with_early_stopping_and_checkpoint(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            IMDB_data_to_predict_vector,
            batch_size=32,
            name="model",
            epochs=100,
        )
    )

    # Plot the loss and categoracal accuracy for each epoch during training
    plot_metric_graph(hist, metric="loss", title="Epoch-Loss Graph")
    plot_metric_graph(
        hist, metric="binary_accuracy", title="Epoch-binary Accuracy Graph"
    )

    # Print the test loss and binary_accuracy of the trained model
    print("################################")
    print("Model Evaluation Metrics:")
    print(f"loss: {loss}\nbinary_accuracy: {binary_accuracy}")
    print("################################")

    # Print the predicted sentiment for each individual in the array
    for prediction in predictions:
        print("Positive" if prediction > 0.5 else "**Negative")


if __name__ == "__main__":
    main()

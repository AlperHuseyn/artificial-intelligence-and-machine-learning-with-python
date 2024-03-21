"""
Sentiment Analysis with IMDB Dataset.

This script performs sentiment analysis on the IMDB movie review dataset using a 2-layer Sequential model in Keras.
It loads the positive and negative movie reviews, preprocesses the data, creates a model, trains it, evaluates it on
a test set, and then predicts the sentiment of new reviews.

Requirements:
- matplotlib
- numpy
- pandas
- scikit-learn
- tensorflow

Make sure to have the `aclImdb` folder in the same directory as this script, containing the train and test folders with
positive and negative reviews.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_IMDB_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for IMDB review-sentiment prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 64 neurons each with a ReLU activation function
    # The final layer has a three neuron with a sigmoid activation function
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(64, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with binary_crossentropy loss function, rmsprop optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=5):
    """
Train, evaluate, and save the IMDB review-sentiment prediction model.

Args:
    X_train (pandas.DataFrame): Input features for training.
    y_train (pandas.Series): Output values for training.
    X_test (pandas.DataFrame): Input features for testing.
    y_test (pandas.Series): Output values for testing.
    X_to_predict (numpy.ndarray): Input features for predictions.
    name (str): Name for saving the model.
    epochs (int): Number of training epochs.

Returns:
    float: binary accuracy of the model on the test dataset.
    numpy.ndarray: Predicted IMDB review-sentiment.
"""
    # Create model using create_IMDB_model func.
    model = create_IMDB_model(input_dim=X_train.shape[1], name='IMDB-review-sentiment')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.2)
    # Evaluate the model on the test dataset
    loss, binary_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict review-sentiment from comments
    predictions = model.predict(X_to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, loss, binary_accuracy, predictions

def plot_epoch_loss_graph(hist, title='Epoch-Loss Graph'):
    """
    Plot the training and validation loss as a function of epochs.

    Args:
        hist (keras.callbacks.History): Training history object.
        title (str): Title for the plot.
    """
    x = hist.epoch
    y = hist.history['loss']
    z = hist.history['val_loss']
    
    # Create plot with custom styling
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=2, color='blue', label='tarining loss')
    ax.plot(x, z, linewidth=2, color='orange', label='validation loss')
    ax.set_title('Epochs vs Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(alpha=.5)
    ax.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
    
def plot_epoch_binary_accuracy_graph(hist, title='Epoch-Binary Accuracy Graph'):
    
    """
   Plot the training and validation Binary Accuracy as a function of epochs.

   Args:
       hist (keras.callbacks.History): Training history object.
       title (str): Title for the plot.
   """
    x = hist.epoch
    y = hist.history['binary_accuracy']
    z = hist.history['val_binary_accuracy']
    
    # Create plot with custom styling
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=2, color='blue', label='tarining binary accuracy')
    ax.plot(x, z, linewidth=2, color='orange', label='validation binary accuracy')
    ax.set_title('Epochs vs binary_accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('binary_accuracy')
    ax.grid(alpha=.5)
    plt.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

def get_subfolders(parent, pos, neg):
    """
    Extract and organize positive and negative subfolders from the parent directory.

    Args:
        parent (str): Path to the parent directory.
        pos (str): Name of the positive subfolder.
        neg (str): Name of the negative subfolder.

    Returns:
        list: List of text data from positive and negative subfolders.
        list: List of corresponding labels (1 for positive, 0 for negative).
    """
    pos_path = None
    neg_path = None
    
    for dirpath, dirnames, filenames in os.walk(parent):
        # Iterate over the subdirectories in the current directory
        for dirname in dirnames:
            if dirname == pos:
                pos_path = os.path.join(dirpath, dirname)
                if neg_path is not None:
                    break  # Stop the iteration if both folders are found
            elif dirname == neg:
                neg_path = os.path.join(dirpath, dirname)
                if pos_path is not None:
                    break  # Stop the iteration if both folders are found
                    
        if pos_path is not None and neg_path is not None:
            break  # Stop the iteration if both folders are found
            
    pos_files = []
    neg_files = []
    
    if pos_path is not None:
        for filename in os.listdir(pos_path):
            file_path = os.path.join(pos_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='latin1') as file:
                    pos_files.append(file.read())
                    y_pos = [1] * len(pos_files)
                    
    if neg_path is not None:
        for filename in os.listdir(neg_path):
            file_path = os.path.join(neg_path, filename)
            if os.path.isfile(file_path):
                with open(file_path,'r', encoding='latin1') as file:
                    neg_files.append(file.read())
                    y_neg = [0] * len(neg_files)
    
    X_dataset = pos_files + neg_files
    y_dataset = y_pos + y_neg
    
    return X_dataset, y_dataset

def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    X_train, y_train = get_subfolders(r'..\aclImdb\train', 'pos', 'neg')    
    X_test, y_test = get_subfolders(r'..\aclImdb\test', 'pos', 'neg')
    
    y_train_ = np.array(y_train)
    y_test_ = np.array(y_test)
    
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer(dtype='uint8', binary=True)
    vectorizer.fit(X_train + X_test)
    X_train_ = vectorizer.transform(X_train).todense()
    X_test_ = vectorizer.transform(X_test).todense()    
    
    # Load the array to be predicted and perform feature scaling on it
    IMDB_data_to_predict = pd.read_csv('predicted.csv')
    
    # Apply vectorization to the data array for prediction
    IMDB_data_to_predict_vector = vectorizer.transform(IMDB_data_to_predict).todense()
    
    # Free up memory
    del X_train, X_test, y_train, y_test, IMDB_data_to_predict
        
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, binary_accuracy, predictions = train_evaluate_save_model(X_train_, y_train_, X_test_, y_test_, IMDB_data_to_predict_vector, name='IMDB-sentiment')
    
    # Plot the loss and binary_accuracy for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_binary_accuracy_graph(hist, title='Epoch-Binary Accuracy Graph')
    
    # Print the test loss and binary_accuracy of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\nbinary_accuracy: {binary_accuracy}')
    print('################################')
    
    # Print the predicted sentiment for each individual in the array
    for prediction in predictions:
        print('Positive' if prediction > .5 else '**Negative')
    
    
if __name__ == '__main__':
    main()
    
    
    

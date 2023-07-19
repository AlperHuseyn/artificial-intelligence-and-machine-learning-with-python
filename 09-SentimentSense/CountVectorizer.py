"""
This script trains and evaluates a machine learning model for sentiment prediction using the IMDB dataset. The main focus of this script is to demonstrate the process 
of vectorizing text data and building a basic neural network model. It does not aim to achieve high performance or accuracy. However, by improving the vectorization 
techniques, such as using more advanced text preprocessing methods or employing pre-trained word embeddings, and optimizing the model architecture and hyperparameters, 
you can aim for better performance and accuracy in sentiment prediction tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    model = Sequential()

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

def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    # Read auto-mpg data from data file as pandas DataFrame
    # first and last column is not necessary  
    IMDB_data = pd.read_csv('IMDB Dataset.csv')
    
    # Encode sentiment column (0 - negative, 1 - positive)
    encoder = LabelEncoder()
    y_dataset = encoder.fit_transform(IMDB_data['sentiment']).astype(np.int8)
    
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer(dtype='uint8', binary=True)
    X_dataset = vectorizer.fit_transform(IMDB_data.iloc[:, 0]).todense()
        
    # Split the dataset into training and test sets using train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)
    
    # Load the array to be predicted and perform feature scaling on it
    IMDB_data_to_predict = pd.read_csv('predicted.csv')
    
    # Apply vectorization to the data array for prediction
    IMDB_data_to_predict_vector = vectorizer.transform(IMDB_data_to_predict).todense()
        
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, binary_accuracy, predictions = train_evaluate_save_model(X_train, y_train, X_test, y_test, IMDB_data_to_predict_vector, name='IMDB-sentiment')
    
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
    
    

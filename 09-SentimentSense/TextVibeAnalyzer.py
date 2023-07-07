import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex
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
    # The first two layers have 128 neurons each with a ReLU activation function
    # The final layer has a three neuron with a sigmoid activation function
    model.add(Dense(128, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(128, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with mse loss function, adam optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=10):
    """
Train, evaluate, and save the IMDB-sentiment prediction model.

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
    numpy.ndarray: Predicted auto-mpg.
"""
    # Create model using create_auto_mpg_model func.
    model = create_IMDB_model(input_dim=X_train.shape[1], name='IMDB-sentiment')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    loss, binary_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict DEATH_EVENT due to heart failure
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
    ax.set_title('Epochs vs mae')
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
    
    # Apple vectorization by not using external module 
    unique_words = set()
    for i in range(len(IMDB_data)):
        words = regex.findall(r'\b[\w\'-]+\b', IMDB_data.iloc[i, 0].lower())
        unique_words.update(words)
    
    word_dictionary = {word: index for index, word in enumerate(unique_words)}
    
    X_dataset = np.zeros((IMDB_data.shape[0], len(word_dictionary)), dtype='int8')
    
    for row, text in enumerate(IMDB_data['review']):
        words = regex.findall(r'\b[\w\'-]+\b', text.lower())
        word_indices = [word_dictionary[word] for word in words]
        X_dataset[row, word_indices] = 1
        
    # Split the dataset into training and test sets using train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=.2)
    
    # Load the array to be predicted and perform feature scaling on it
    IMDB_data_to_predict = pd.read_csv('predicted.csv').to_numpy()
    
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, binary_accuracy, predictions = train_evaluate_save_model(X_train, y_train, X_test, y_test, IMDB_data_to_predict, name='IMDB-sentiment')
    
    # Plot the loss and mae for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_binary_accuracy_graph(hist, title='Epoch-Binary Accuracy Graph')
    
    # Print the test loss and mae of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\nbinary_accuracy: {binary_accuracy}')
    print('################################')
    
    # Print the predicted mpg for each individual in the array
    for prediction in predictions[:, 0]:
        print(prediction)
    
    
if __name__ == '__main__':
    main()
    
    
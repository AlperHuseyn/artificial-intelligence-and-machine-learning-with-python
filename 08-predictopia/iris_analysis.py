import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_iris_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for iris prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential()

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 64 neurons each with a ReLU activation function
    # The final layer has a three neuron with a softmax activation function
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(64, activation='relu', name='Hidden2'))
    model.add(Dense(3, activation='softmax', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with mse loss function, adam optimizer, and categorical_accuracy metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=100):
    """
Train, evaluate, and save the iris prediction model.

Args:
    X_train (pandas.DataFrame): Input features for training.
    y_train (pandas.Series): Output values for training.
    X_test (pandas.DataFrame): Input features for testing.
    y_test (pandas.Series): Output values for testing.
    X_to_predict (numpy.ndarray): Input features for predictions.
    name (str): Name for saving the model.
    epochs (int): Number of training epochs.

Returns:
    float: Categorical accuracy of the model on the test dataset.
    numpy.ndarray: Predicted auto-mpg.
"""
    # Create model using create_auto_mpg_model func.
    model = create_iris_model(input_dim=X_train.shape[1], name='iris')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    loss, categorical_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict DEATH_EVENT due to heart failure
    predictions = model.predict(X_to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, loss, categorical_accuracy, predictions

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
    
def plot_epoch_categorical_accuracy_graph(hist, title='Epoch-Categorical Accuracy Graph'):
    """
   Plot the training and validation Categorical Accuracy as a function of epochs.

   Args:
       hist (keras.callbacks.History): Training history object.
       title (str): Title for the plot.
   """
    x = hist.epoch
    y = hist.history['categorical_accuracy']
    z = hist.history['val_categorical_accuracy']
    
    # Create plot with custom styling
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=2, color='blue', label='tarining categorical accuracy')
    ax.plot(x, z, linewidth=2, color='orange', label='validation categorical accuracy')
    ax.set_title('Epochs vs mae')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('categorical_accuracy')
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
    iris_data = pd.read_csv('Iris.csv')
    classes = iris_data.iloc[:, -1].unique()
    
    # Apply one-hot encoding to output column (Species)
    iris_data = pd.get_dummies(iris_data, columns=['Species'])
    
    # Split the dataset into training and test sets using train_test_split function
    training_data, test_set = train_test_split(iris_data, test_size=.2)
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = training_data.iloc[:, 1:-3]
    y_train = training_data.iloc[:, -3:]
    
    # Separate the input features (X_test) and output values (y_test) of the training dataset
    X_test = test_set.iloc[:, 1:-3]
    y_test = test_set.iloc[:, -3:]
    
    # Perform feature scaling on the input features
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)    
    
    # Load the array to be predicted and perform feature scaling on it
    iris_data_to_predict = pd.read_csv('predicted.csv').to_numpy()
    scaled_to_predict = scaler.transform(iris_data_to_predict)
    
    # Train and evaluate the machine learning model using the scaled training and test data
    hist, loss, categorical_accuracy, predictions = train_evaluate_save_model(scaled_X_train, y_train, scaled_X_test, y_test, scaled_to_predict, name='iris')
    
    # Plot the loss and mae for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_categorical_accuracy_graph(hist, title='Epoch-Categorical Accuracy Graph')
    
    # Print the test loss and mae of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\ncategorical_accuracy: {categorical_accuracy}')
    print('################################')
    
    # Print the predicted mpg for each individual in the array
    for prediction in classes[np.argmax(predictions, axis=1)]:
        print(prediction)
    
    with open('iris.pickle', 'wb') as f:
        pickle.dump(scaler, f)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
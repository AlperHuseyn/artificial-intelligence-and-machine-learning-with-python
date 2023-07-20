import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def create_auto_mpg_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for auto-mpg prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 64 neurons each with a ReLU activation function
    # The final layer has a single neuron with a linear activation function
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(64, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='linear', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with mse loss function, rmsprop optimizer, and mae metrics
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=100):
    """
Train, evaluate, and save the auto-mpg prediction model.

Args:
    X_train (pandas.DataFrame): Input features for training.
    y_train (pandas.Series): Output values for training.
    X_test (pandas.DataFrame): Input features for testing.
    y_test (pandas.Series): Output values for testing.
    X_to_predict (numpy.ndarray): Input features for predictions.
    name (str): Name for saving the model.
    epochs (int): Number of training epochs.

Returns:
    float: Mean Absolute Error (MAE) of the model on the test dataset.
    numpy.ndarray: Predicted auto-mpg.
"""
    # Create model using create_auto_mpg_model func.
    model = create_auto_mpg_model(input_dim=X_train.shape[1], name='auto-mpg')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    # Predict DEATH_EVENT due to heart failure
    predictions = model.predict(X_to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, loss, mae, predictions

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
    
def plot_epoch_mae_graph(hist, title='Epoch-MAE Graph'):
    """
   Plot the training and validation Mean Absolute Error (MAE) as a function of epochs.

   Args:
       hist (keras.callbacks.History): Training history object.
       title (str): Title for the plot.
   """
    x = hist.epoch
    y = hist.history['mae']
    z = hist.history['val_mae']
    
    # Create plot with custom styling
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=2, color='blue', label='tarining mae')
    ax.plot(x, z, linewidth=2, color='orange', label='validation mae')
    ax.set_title('Epochs vs mae')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mae')
    ax.grid(alpha=.5)
    plt.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
    
def main():
    """
    Main function to train and evaluate the auto-mpg prediction model.
    """
    # Define the divide ratio for splitting the dataset
    DIVIDE_RATIO = .8
    
    # Read auto-mpg data from data file as pandas DataFrame
    # last column is not necessary  
    auto_mpg_data = pd.read_csv('auto-mpg.data', delimiter=r'\s+', header=None).iloc[:,:-1]  
    
    # get rid of the missing values at 3rd column
    auto_mpg_data = auto_mpg_data[auto_mpg_data.iloc[:, 3] != '?']
    
    # Apply ohe to 7th column (categorical)
    auto_mpg_data = pd.get_dummies(auto_mpg_data, columns=[7]).to_numpy(dtype='float32') 
    
    training_set, test_set = auto_mpg_data[:, 1:], auto_mpg_data[:, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(training_set, test_set, test_size=1-DIVIDE_RATIO)
    
    scaler = MinMaxScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)    

    # Load the array to be predicted and perform feature scaling on it
    auto_mpg_data_to_predict = pd.read_csv('predicted.csv').to_numpy()
    scaled_to_predict = scaler.transform(auto_mpg_data_to_predict)

    # Train and evaluate the machine learning model using the scaled training and test data
    hist, loss, mae, predictions = train_evaluate_save_model(scaled_X_train, y_train, scaled_X_test, y_test, scaled_to_predict, name='auto-mpg')
    
    # Plot the loss and mae for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_mae_graph(hist, title='Epoch-MAE Graph')
    
    # Print the test loss and mae of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\nMAE: {mae}')
    print('################################')
    
    # Print the predicted mpg for each individual in the array
    for prediction in predictions[:, 0]:
        print(prediction)
    
    with open('auto_mpg.pickle', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()

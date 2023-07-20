import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle


def create_2L_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for heart failure prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 16 neurons each with a ReLU activation function
    # The final layer has a single neuron with a sigmoid activation function
    model.add(Dense(16, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(16, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compile the model with binary cross-entropy loss function, adam optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, to_predict, epochs=100, name='model'):
    """
    Train and evaluate the heart failure prediction model.

    Args:
        X_train (pandas.DataFrame): Input features for training.
        y_train (pandas.Series): Output values for training.
        X_test (pandas.DataFrame): Input features for prediction.
        y_test (pandas.Series): Output values for prediction.
        to_predict (numpy.ndarray): Input features for which predictions are to be made.
        epochs (int): Number of training epochs.
        

    Returns:
        float: Accuracy of the model on the test dataset.
        numpy.ndarray: Predicted heart failure risk for the X_test dataset.
    """
    # Create model using create_2L_model func.
    model = create_2L_model(input_dim=X_train.shape[1], name='heart-Failure-Predictor')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict DEATH_EVENT due to heart failure
    predictions = model.predict(to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, accuracy, predictions

def plot_epoch_loss_graph(hist, title='Epoch-Loss Graph'):
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
    
def plot_epoch_accuracy_graph(hist, title='Epoch-Accuracy Graph'):
    x = hist.epoch
    y = hist.history['binary_accuracy']
    z = hist.history['val_binary_accuracy']
    
    # Create plot with custom styling
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=2, color='blue', label='tarining accuracy')
    ax.plot(x, z, linewidth=2, color='orange', label='validation accuracy')
    ax.set_title('Epochs vs Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=.5)
    plt.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

def main():
    """
    Main function to train and evaluate the heart failure prediction model.
    """
    # Define the divide ratio for splitting the dataset
    DIVIDE_RATIO = .9
    
    # Read heart failure data from csv file as pandas DataFrame
    heart_failure_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # Split the dataset into training and test sets using train_test_split function
    training_data, test_set = train_test_split(heart_failure_data, test_size=1-DIVIDE_RATIO)
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = training_data.iloc[:, :-1]
    y_train = training_data.iloc[:, -1]
    
    # Separate the input features (X_test) and output values (y_test) of the training dataset
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    
    # Perform feature scaling on the input features using MinMaxScaler
    scaler = MinMaxScaler()
    feature_scaled_training_data = scaler.fit_transform(X_train)
    feature_scaled_test_data = scaler.transform(X_test)
    
    # Load the array to be predicted and perform feature scaling on it
    heart_failure_data_to_predict = pd.read_csv('predicted.csv').to_numpy()
    feature_scaled_to_predict = scaler.transform(heart_failure_data_to_predict)
    
    # Train and evaluate the machine learning model using the scaled training and test data
    hist, accuracy, predictions = train_and_evaluate_model(feature_scaled_training_data, y_train, feature_scaled_test_data, y_test, feature_scaled_to_predict, name='heart-failure')
    
    # Plot the loss and accuracy for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph_')    
    plot_epoch_accuracy_graph(hist, title='Epoch-Accuracy Graph_')
    
    # Print the test accuracy of the trained model
    print('################################')
    print(f'Test accuracy: {accuracy}')
    print('################################')
    
    # Print the predicted risk of heart failure for each individual in the array
    for prediction in predictions:
        print('*heart failure risk is high...' if prediction > .5 else 'Person has a low risk of heart failure...')
    
    with open('heart-failure.pickle', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()

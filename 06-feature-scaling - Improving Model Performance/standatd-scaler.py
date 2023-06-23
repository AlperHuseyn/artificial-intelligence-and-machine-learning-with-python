import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def create_2L_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for heart failure prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential()

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 16 neurons each with a ReLU activation function
    # The final layer has a single neuron with a sigmoid activation function
    model.add(Dense(16, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(16, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compile the model with binary cross-entropy loss function, adam optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    return model

def train_and_evaluate_model(regressor, regressor_outputs, predictor, predictor_outputs, epochs=100):
    """
    Train and evaluate the heart failure prediction model.

    Args:
        regressor (pandas.DataFrame): Input features for training.
        regressor_outputs (pandas.Series): Output values for training.
        predictor (pandas.DataFrame): Input features for prediction.
        predictor_outputs (pandas.Series): Output values for prediction.
        to_predict (numpy.ndarray): Input features for which predictions are to be made.
        epochs (int): Number of training epochs.
        

    Returns:
        float: Accuracy of the model on the test dataset.
        numpy.ndarray: Predicted heart failure risk for the predictor dataset.
    """
    # Create model using create_2L_model func.
    model = create_2L_model(input_dim=regressor.shape[1], name='heart-Failure-Predictor')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(regressor, regressor_outputs, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    model.evaluate(predictor, predictor_outputs, verbose=0)
        
    return hist

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
    
    # Separate the input features (regressor) and output values (regressor_outputs) of the training dataset
    regressor = training_data.iloc[:, :-1]
    regressor_outputs = training_data.iloc[:, -1]
    
    # Separate the input features (predictor) and output values (predictor_outputs) of the training dataset
    predictor = test_set.iloc[:, :-1]
    predictor_outputs = test_set.iloc[:, -1]
    
    standard_scaler = StandardScaler()
    standard_scaler.fit(regressor)
    feature_scaled = standard_scaler.transform(regressor)
    
    hist = train_and_evaluate_model(feature_scaled, regressor_outputs, predictor, predictor_outputs)
    
    plot_epoch_loss_graph(hist)
    
    plot_epoch_accuracy_graph(hist)
    
if __name__ == '__main__':
    main()

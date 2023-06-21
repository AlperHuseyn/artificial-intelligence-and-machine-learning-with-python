import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import time


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
    # The first two layers have 32 neurons each with a ReLU activation function
    # The final layer has a single neuron with a sigmoid activation function
    model.add(Dense(16, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(16, activation='relu', name='Hidden2'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Compile the model with binary cross-entropy loss function, adam optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    return model

def train_and_evaluate_model(regressor, regressor_outputs, predictor, predictor_outputs, to_predict, epochs=100):
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
    model.fit(regressor, regressor_outputs, epochs=epochs, validation_split=.1)
    # Evaluate the model on the test dataset
    _, accuracy = model.evaluate(predictor, predictor_outputs, verbose=0)
    # Predict DEATH_EVENT due to heart failure
    predictions = model.predict(to_predict)
    
    return accuracy, predictions


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
    
    # Load the array to be predicted
    heart_failure_data_to_predict = pd.read_csv('predicted.csv').to_numpy()

    for epoch in [1, 50, 100, 150, 200]:
        accuracy, predictions = train_and_evaluate_model(regressor, regressor_outputs, predictor, predictor_outputs, heart_failure_data_to_predict, epochs=epoch)
        
        with open(f'epoch-value-{epoch}.txt', 'w', encoding='utf-8') as f:
            # Print the accuracy of the model on the test dataset
            print(f'Test accuracy: {accuracy}', end='\n\n', file=f)
            
            # Print predictions into a txt file
            for prediction in predictions:
                print('heart failure risk is high...' if prediction > .5 else 'Person has a low risk of heart failure...', file=f)
                
        time.sleep(10)
        

if __name__ == '__main__':
    main()

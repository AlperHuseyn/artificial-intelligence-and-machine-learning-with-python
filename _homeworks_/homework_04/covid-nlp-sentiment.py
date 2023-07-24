import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_covid_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for covid-nlp prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 64 neurons each with a ReLU activation function
    # The final layer has a three neuron with a softmax activation function
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(64, activation='relu', name='Hidden2'))
    model.add(Dense(5, activation='softmax', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with binary_crossentropy loss function, rmsprop optimizer, and binary_accuracy metrics
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=5):
    """
Train, evaluate, and save the covid-nlp prediction model.

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
    # Create model using create_reuters_model func.
    model = create_covid_model(input_dim=X_train.shape[1], name='Covid-nlp')
    # Train the model on the training dataset
    # Use 10% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.2)
    # Evaluate the model on the test dataset
    loss, category_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict review-sentiment from comments
    predictions = model.predict(X_to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, loss, category_accuracy, predictions

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
    ax.set_title('Epochs vs categorical_accuracy')
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
    # Read tweeter training data from data file as pandas DataFrame
    tweeter_training_data = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1').dropna(subset=['OriginalTweet', 'Sentiment'])
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = tweeter_training_data['OriginalTweet'].tolist()
    y_train = tweeter_training_data.iloc[:, -1].astype(str)
    
    # Convert labels to a NumPy array for indexing purposes
    labels = np.array(np.unique(y_train))
    
    # Read tweeter test data from data file as pandas DataFrame
    tweeter_test_data = pd.read_csv('Corona_NLP_test.csv', encoding='latin-1')
    
    # Separate the input features (X_test) and output values (y_test) of the test dataset
    X_test = tweeter_test_data['OriginalTweet'].tolist()
    y_test = tweeter_test_data.iloc[:, -1].astype(str)
    
    # One Hot Encode y_train and y_test
    encoder = OneHotEncoder(sparse_output=False, dtype='uint8')
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1))
    
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer(dtype='uint8')
    vectorizer.fit(X_train + X_test)
    X_train = vectorizer.transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
    
    # Load the array to be predicted 
    covid_data_to_predict = pd.read_csv('predicted.csv')['OriginalTweet'].tolist()
    
    # Apply vectorization to the data array for prediction
    covid_data_to_predict_vector = vectorizer.transform(covid_data_to_predict).todense()
    
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, categorical_accuracy, predictions = train_evaluate_save_model(X_train, y_train, X_test, y_test, covid_data_to_predict_vector, name='covid-sentiment')
    
    # Free up memory
    del tweeter_test_data, tweeter_training_data, covid_data_to_predict
    
    # Plot the loss and mae for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_categorical_accuracy_graph(hist, title='Epoch-Categorical Accuracy Graph')
    
    # Print the test loss and mae of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\ncategorical_accuracy: {categorical_accuracy}')
    print('################################')
       
    # Print the predicted label for each individual in the array
    for prediction in np.argmax(predictions, axis=1):
        print(labels[prediction])

if __name__ == '__main__':
    main()
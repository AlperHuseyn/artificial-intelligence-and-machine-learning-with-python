import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_reuters_model(input_dim, name=None):
    """
    Create a 2-layer Sequential model for reuters review-sentiment prediction.

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
    model.add(Dense(81, activation='softmax', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with binary_crossentropy loss function, rmsprop optimizer, and binary_accuracy metrics
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, name='model', epochs=5):
    """
Train, evaluate, and save the reuters review-sentiment prediction model.

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
    model = create_reuters_model(input_dim=X_train.shape[1], name='Reuters-review-sentiment')
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
    PATH = os.getcwd()
    
    # Get labels from cats.txt file for both training and test sets
    training_labels = {}
    test_labels = {}
    
    with open(f'{PATH}/reuters/cats.txt', encoding='latin-1') as file:
        for line in file:
            match = re.match(r'(training|test)/(\d+)\s+(\w+)', line)
            group, num, label = match[1], match[2], match[3]
            if group == 'training':
                training_labels[int(num)] = label
            elif group == 'test':
                test_labels[int(num)] = label
                
    labels = set(training_labels.values())
    labels.update(test_labels.values())
    labels = list(labels)
    
    X_train = []
    y_train = []
    
    for fname in os.listdir(f'{PATH}/reuters/training'):
        with open(f'{PATH}/reuters/training/' + fname, encoding='latin-1') as file:
            X_train.append(file.read())
        lab = training_labels[int(fname)]
        idx = labels.index(lab)
        y_train.append(idx)
    
    X_test = []
    y_test = []
    
    for fname in os.listdir(f'{PATH}/reuters/test'):
        with open(f'{PATH}/reuters/test/' + fname, encoding='latin-1') as file:
            X_test.append(file.read())
        lab = test_labels[int(fname)]
        idx = labels.index(lab)
        y_test.append(idx)
        
    # One Hot Encode y_train and y_test
    encoder = OneHotEncoder(sparse_output=False, categories=[range(len(labels))])
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1))
        
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train + X_test)
    X_train = vectorizer.transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
    
    # Load the array to be predicted and perform feature scaling on it
    reuters_data_to_predict = pd.read_csv('reuters-predicted.csv')['text'].tolist()
    
    # Apply vectorization to the data array for prediction
    reuters_data_to_predict_vector = vectorizer.transform(reuters_data_to_predict).todense()
    
    # Free up memory
    del reuters_data_to_predict, file, fname, group, idx, lab, label,  line, match, num, PATH, test_labels, training_labels 
    
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, categorical_accuracy, predictions = train_evaluate_save_model(X_train, y_train, X_test, y_test, reuters_data_to_predict_vector, name='reuters-sentiment')
    
    # Plot the loss and mae for each epoch during training
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_categorical_accuracy_graph(hist, title='Epoch-Categorical Accuracy Graph')
    
    # Print the test loss and mae of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\ncategorical_accuracy: {categorical_accuracy}')
    print('################################')
    
    # Convert labels to a NumPy array for indexing purposes
    labels = np.array(labels)
    
    # Print the predicted label for each individual in the array
    for prediction in np.argmax(predictions, axis=1):
        print(labels[prediction])
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

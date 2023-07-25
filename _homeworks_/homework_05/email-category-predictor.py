import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_email_model(input_dim, num_categories, name=None):
    """
    Create a 2-layer Sequential model for email prediction.

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
    model.add(Dense(num_categories, activation='softmax', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with categorical_crossentropy loss function, rmsprop optimizer, and categorical_accuracy metrics
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_test, y_test, X_to_predict, num_categories, name='model', epochs=5):
    """
Train, evaluate, and save the email prediction model.

Args:
    X_train (pandas.DataFrame): Input features for training.
    y_train (pandas.Series): Output values for training.
    X_test (pandas.DataFrame): Input features for testing.
    y_test (pandas.Series): Output values for testing.
    X_to_predict (numpy.ndarray): Input features for predictions.
    name (str): Name for saving the model.
    epochs (int): Number of training epochs.

Returns:
    float: categorical accuracy of the model on the test dataset.
    numpy.ndarray: Predicted email-category.
"""
    # Create model using create_email_model func.
    model = create_email_model(input_dim=X_train.shape[1], num_categories=num_categories, name='email-category-predictor')
    # Train the model on the training dataset
    # Use 20% of the training data as validation data to monitor the model's performance during training
    # The 'hist' object contains training history, which is used to plot an epoch-loss graph to determine the optimal number of epochs and avoid overfitting.
    hist = model.fit(X_train, y_train, epochs=epochs, validation_split=.2)
    # Evaluate the model on the test dataset
    loss, category_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Predict email-category from sent categorized-emails
    predictions = model.predict(X_to_predict)
    
    model.save(f'{name}.h5')
        
    return hist, loss, category_accuracy, predictions

def plot_metric_graph(hist, metric, title):
    """
    Plot the training and validation metric as a function of epochs.

    Args:
        hist (keras.callbacks.History): Training history object.
        metric (str): Name of the metric to plot.
        title (str): Title for the plot.
    """
    # Create plot with custom styling
    plt.figure(figsize=(15, 5))
    plt.plot(hist.epoch, hist.history[metric], linewidth=2, color='blue', label=f'training {metric}')
    plt.plot(hist.epoch, hist.history[f'val_{metric}'], linewidth=2, color='orange', label=f'validation {metric}')
    plt.title(f'Epochs vs {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.grid(alpha=0.5)
    plt.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
    
def get_subfolders(parent, category_folders):
    """
    Extract and organize subfolders from the parent directory.

    Args:
        parent (str): Path to the parent directory.
        category_folders (list): List of category subfolders.

    Returns:
        pandas.DataFrame: Concatenated data from the category subfolders.
    """
    data = []
    for category in category_folders:
        category_path = os.path.join(parent, category)
        if os.path.exists(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        data.append({'Emails': file.read(), 'Category': category})
    return pd.DataFrame(data)

def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    # Read email data from folders as pandas DataFrame
    email_data = get_subfolders(r'Data', ['Crime', 'Entertainment', 'Politics', 'Science'])
    
    # Split the dataset into training and test sets using train_test_split function
    training_set, test_set = train_test_split(email_data, test_size=.2)
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = training_set['Emails'].tolist()
    y_train = training_set['Category']
    
    # Separate the input features (X_test) and output values (y_test) of the training dataset
    X_test = test_set['Emails'].tolist()
    y_test = test_set['Category']
    
    # Convert labels to a NumPy array for indexing purposes
    labels = np.array(np.unique(y_train))
    
    # One Hot Encode y_train and y_test
    encoder = OneHotEncoder(sparse_output=False, dtype='uint8')
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1))
    
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer(dtype='uint8', binary=True)
    vectorizer.fit(X_train + X_test)
    X_train = vectorizer.transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
    
    # Load the array to be predicted 
    email_data_to_predict = pd.read_csv('email-predicted.csv')['Emails'].tolist()
    
    # Apply vectorization to the data array for prediction
    email_data_to_predict_vector = vectorizer.transform(email_data_to_predict).todense()
    
    # Get the number of output categories (which is the number of unique labels)
    num_categories = len(labels)
    
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, categorical_accuracy, predictions = train_evaluate_save_model(X_train, y_train, X_test, y_test, email_data_to_predict_vector, num_categories=num_categories, name='email-category-predictor')
    
    # Free up memory
    del email_data, email_data_to_predict, test_set, training_set
    
    # Plot the loss and categoracal accuracy for each epoch during training
    plot_metric_graph(hist, metric='loss', title='Epoch-Loss Graph')    
    plot_metric_graph(hist, metric='categorical_accuracy', title='Epoch-Categorical Accuracy Graph')
    
    # Print the test loss and categorical accuracy of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\ncategorical_accuracy: {categorical_accuracy}')
    print('################################')
       
    # Print the predicted label for each individual in the array
    for prediction in np.argmax(predictions, axis=1):
        print(labels[prediction])
    

if __name__ == '__main__':
    main()
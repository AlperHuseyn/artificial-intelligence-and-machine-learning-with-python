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

def get_subfolders(parent, crime, entertainment, politics, science):
    """
    Extract and organize crime, entertainment, politics, and science  subfolders from the parent directory.

    Args:
        parent (str): Path to the parent directory.
        crime (str): Name of the crime subfolder.
        entertainment (str): Name of the entertainment subfolder.
        politics (str): Name of the politics subfolder.
        science (str): Name of the science subfolder.
        
    Returns:
        list: List of text data from crime, entertainment, politics, and science subfolders.
        list: List of corresponding labels.
    """
    crime_path = None
    entertainment_path = None
    politics_path = None
    science_path = None
    
    for dirpath, dirnames, filenames in os.walk(parent):
        # Iterate over the subdirectories in the current directory
        for dirname in dirnames:
            if dirname == crime:
                crime_path = os.path.join(dirpath, dirname)
                if entertainment_path is not None and politics_path is not None and science_path is not None:
                    break  # Stop the iteration if both folders are found
            elif dirname == entertainment:
                entertainment_path = os.path.join(dirpath, dirname)
                if crime_path is not None and politics_path is not None and science_path is not None:
                    break  # Stop the iteration if both folders are found
            elif dirname == politics:
                politics_path = os.path.join(dirpath, dirname)
                if crime_path is not None and entertainment_path is not None and science_path is not None:
                    break  # Stop the iteration if both folders are found
            elif dirname == science:
                science_path = os.path.join(dirpath, dirname)
                if crime_path is not None and entertainment_path is not None and politics_path is not None:
                    break  # Stop the iteration if both folders are found
                    
        if crime_path is not None and entertainment_path is not None and politics_path is not None and science_path is not None:
            break  # Stop the iteration if both folders are found
            
    crime_files = []
    entertainment_files = []
    politics_files = []
    science_files = []
    
    if crime_path is not None:
        for filename in os.listdir(crime_path):
            file_path = os.path.join(crime_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='latin1') as file:
                    crime_files.append(file.read())
                    y_crime = [f'{crime}'] * len(crime_files)
        crime_dataframe = pd.DataFrame(data={'Emails': crime_files, 'Category': y_crime}) 
                    
    if entertainment_path is not None:
        for filename in os.listdir(entertainment_path):
            file_path = os.path.join(entertainment_path, filename)
            if os.path.isfile(file_path):
                with open(file_path,'r', encoding='latin1') as file:
                    entertainment_files.append(file.read())
                    y_entertainment = [f'{entertainment}'] * len(entertainment_files)
        entertainment_dataframe = pd.DataFrame(data={'Emails': entertainment_files, 'Category': y_entertainment}) 
                    
    if politics_path is not None:
        for filename in os.listdir(politics_path):
            file_path = os.path.join(politics_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='latin1') as file:
                    politics_files.append(file.read())
                    y_politics = [f'{politics}'] * len(politics_files)
        politics_dataframe = pd.DataFrame(data={'Emails': politics_files, 'Category': y_politics}) 
                    
    if science_path is not None:
        for filename in os.listdir(science_path):
            file_path = os.path.join(science_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='latin1') as file:
                    science_files.append(file.read())
                    y_science = [f'{science}'] * len(science_files)
        science_dataframe = pd.DataFrame(data={'Emails': science_files, 'Category': y_science}) 
    
    dataset = pd.concat([crime_dataframe, entertainment_dataframe, politics_dataframe, science_dataframe])
    
    return dataset

def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    # Read email data from folders as pandas DataFrame
    email_data = get_subfolders(r'Data', 'Crime', 'Entertainment', 'Politics', 'Science')
    
    # Split the dataset into training and test sets using train_test_split function
    training_set, test_set = train_test_split(email_data, test_size=.2)
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = training_set['Emails'].tolist()
    y_train = training_set.iloc[:, -1]
    
    # Separate the input features (X_test) and output values (y_test) of the training dataset
    X_test = test_set['Emails'].tolist()
    y_test = test_set.iloc[:, -1]
    
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
    plot_epoch_loss_graph(hist, title='Epoch-Loss Graph')    
    plot_epoch_categorical_accuracy_graph(hist, title='Epoch-Categorical Accuracy Graph')
    
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
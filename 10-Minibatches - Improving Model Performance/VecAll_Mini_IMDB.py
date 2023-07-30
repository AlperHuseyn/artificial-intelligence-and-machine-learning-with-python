import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence


class CustomDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, shuffle=True):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.data.shape[0])  # sparse matrix has no len function, used shape[0] instead
        
    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))
    
    def __getitem__(self, index):        
        start_idx = self.indices[index] * self.batch_size 
        stop_idx = (self.indices[index] + 1) * self.batch_size
                
        # Read the data in batches
        batch_data = self.data[start_idx:stop_idx].toarray()
        batch_labels = self.labels[start_idx:stop_idx]
        
        return batch_data, np.array(batch_labels)
    
    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_IMDB_model(input_dim, num_categories, name=None):
    """
    Create a 2-layer Sequential model for IMDB review-sentiment prediction.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Sequential: Created model.
    """
    model = Sequential(name=name)

    # Define the architecture of the neural network by adding layers to the model
    # The first two layers have 64 neurons each with a ReLU activation function
    # The final layer has a three neuron with a sigmoid activation function
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden1'))
    model.add(Dense(64, activation='relu', name='Hidden2'))
    model.add(Dense(num_categories, activation='sigmoid', name='output'))
    
    # Print model info on console
    model.summary()

    # Compile the model with binary_crossentropy loss function, rmsprop optimizer, and binary_accuracy metrics
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
    
    return model

def train_evaluate_save_model(X_train, y_train, X_valid, y_valid, X_test, y_test, vectorizer, num_categories, X_to_predict, batch_size=32, name='model', epochs=5):
    """
    Train, evaluate, and save the IMDB review-sentiment prediction model.

    Args:
        X_train (pandas.DataFrame): Input features for training.
        y_train (pandas.Series): Output values for training.
        X_valid (pandas.DataFrame): Input features for validation.
        y_valid (pandas.Series): Output values for validation.
        X_test (pandas.DataFrame): Input features for testing.
        y_test (pandas.Series): Output values for testing.
        batch_size (int): Batch size for training and evaluation.
        name (str): Name for saving the model.
        epochs (int): Number of training epochs.

    Returns:
        float: Binary accuracy of the model on the test dataset.
        numpy.ndarray: Predicted IMDB review-sentiment.
    """
    # Create model using create_IMDB_model func.
    model = create_IMDB_model(input_dim=len(vectorizer.vocabulary_), num_categories=num_categories, name='IMDB-review-sentiment')
    
    # Create custom data generators for training, validation, and testing
    train_data_generator = CustomDataGenerator(X_train, y_train, batch_size)
    valid_data_generator = CustomDataGenerator(X_valid, y_valid, batch_size, shuffle=False)
    test_data_generator = CustomDataGenerator(X_test, y_test, batch_size, shuffle=False)    
    pred_data_generator = CustomDataGenerator(X_to_predict, pd.Series(dtype='uint8'), batch_size)
    
    # Train the model using the custom data generator for training, test, and validation data
    hist = model.fit(train_data_generator, epochs=epochs, validation_data=valid_data_generator)
    print()
    # Evaluate the model on the test dataset
    loss, binary_accuracy = model.evaluate(test_data_generator)
    print()
    # Predict review-sentiment from comments
    predictions = model.predict(pred_data_generator)
    print()
    
    model.save(f'{name}.h5')
        
    return hist, loss, binary_accuracy, predictions

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

def main():
    """
    Main function to train and evaluate the iris prediction model.
    """
    # Read IMDB data from data file as pandas DataFrame
    IMDB_data = pd.read_csv('IMDB Dataset.csv')
    
    # Encode sentiment column (0 - negative, 1 - positive)
    encoder = LabelEncoder()
    IMDB_data['sentiment'] = encoder.fit_transform(IMDB_data['sentiment']).astype(np.int8)
    
    # Split the dataset into training and test sets using train_test_split function
    training_set, test_data = train_test_split(IMDB_data)
    
    # Split the trainin_set into training_data and validation_data using train_test_split function
    training_data, validation_data = train_test_split(training_set)
    
    # Separate the input features (X_train) and output values (y_train) of the training dataset
    X_train = training_data['review']
    y_train = training_data.iloc[:, -1]

    # Separate the input features (X_test) and output values (y_test) of the test dataset
    X_test = test_data['review']
    y_test = test_data.iloc[:, -1]
    
    # Separate the input features (X_valid) and output values (y_valid) of the training dataset
    X_valid = validation_data['review']
    y_valid = validation_data.iloc[:, -1]
    
    # Apply vectorization to convert text into a numerical representation
    vectorizer = CountVectorizer(dtype='uint8', binary=True)
    vectorizer.fit(pd.concat([X_train, X_valid, X_test]))
   
    # Convert text data to numerical representations using the CountVectorizer
    X_train_vector = vectorizer.transform(X_train)
    X_valid_vector = vectorizer.transform(X_valid)
    X_test_vector = vectorizer.transform(X_test)
    
    # Load the array to be predicted and perform feature scaling on it
    IMDB_data_to_predict = pd.read_csv('predicted.csv')['review'].tolist()

    # Apply vectorization to the data array for prediction
    IMDB_data_to_predict_vector = vectorizer.transform(IMDB_data_to_predict)
    
    # 0 for negatives and 1 for positives
    num_categories = 1
    
    # Train and evaluate the machine learning model using the training and test data
    hist, loss, binary_accuracy, predictions = train_evaluate_save_model(X_train_vector, y_train, X_valid_vector, y_valid, X_test_vector, y_test, vectorizer, num_categories, IMDB_data_to_predict_vector, name='IMDB-sentiment')
    
    # Free up memory
    del IMDB_data, training_set, X_train, X_train_vector, X_test, X_test_vector, X_valid, X_valid_vector, y_valid, y_train, y_test, IMDB_data_to_predict        
    
    # Plot the loss and categoracal accuracy for each epoch during training
    plot_metric_graph(hist, metric='loss', title='Epoch-Loss Graph')    
    plot_metric_graph(hist, metric='binary_accuracy', title='Epoch-Binary Accuracy Graph')
    
    # Print the test loss and binary_accuracy of the trained model
    print('################################')
    print("Model Evaluation Metrics:")
    print(f'loss: {loss}\nbinary_accuracy: {binary_accuracy}')
    print('################################')
    
    # Print the predicted sentiment for each individual in the array
    for prediction in predictions:
        print('Positive' if prediction > .5 else '**Negative**')
    
    
if __name__ == '__main__':
    main()
    

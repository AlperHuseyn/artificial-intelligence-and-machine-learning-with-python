"""
This code builds a neural network model to predict diabetes based on input features. 
While this code provides a general idea of how to build a neural network model using Keras, 
it may not necessarily be the best solution for predicting diabetes with high accuracy.

It is important to note that building an accurate predictive model requires careful consideration 
of various factors such as data quality, preprocessing, feature selection, hyperparameter tuning, 
and model evaluation. Therefore, before applying this code to real-world applications, further 
modifications and optimizations may be necessary to improve its performance.

Overall, this code serves as a starting point for understanding the basic structure of a neural 
network model and provides a foundation for further exploration and experimentation in machine learning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Define the divide ratio for splitting the dataset
DIVIDE_RATIO = .8

# Read diabetes data from csv file as pandas DataFrame
diabetes_data = pd.read_csv('diabetes.csv')

# Split the dataset into training and test sets using train_test_split function
train_dataset, test_set = train_test_split(diabetes_data, test_size=1-DIVIDE_RATIO)

# Separate the input features (regressor) and output values (regressor_outputs) of the training dataset
regressor = train_dataset.iloc[:, :-1]
regressor_outputs = train_dataset.iloc[:, -1]

# Separate the input features (predictor) and output values (predictor_outputs) of the test dataset
predictor = test_set.iloc[:, :-1]
predictor_outputs = test_set.iloc[:, -1]

# Create Sequential model object
model = Sequential(name='Pima-Indians-Diabetes-Predictor')

# Define the architecture of the neural network by adding layers to the model
# The first two layers have 64 neurons each with a ReLU activation function
# The final layer has a single neuron with a sigmoid activation function
model.add(Dense(64, activation='relu', input_dim=regressor.shape[1], name='hidden-1'))
model.add(Dense(64, activation='relu', name='hidden-2'))
model.add(Dense(1, activation='sigmoid', name='output'))

# Print model summary
model.summary()

# Compile the model with binary cross-entropy loss function, adam optimizer, and binary_accuracy metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train the model on the training dataset
# Use 20% of the training data as validation data to monitor the model's performance during training
model.fit(regressor, regressor_outputs, epochs=100, batch_size=32, validation_split=.2)

# Evaluate the model on the test dataset
score = model.evaluate(predictor, predictor_outputs, verbose=0)

# Print the accuracy of the model on the test dataset
print('Test accuracy:', score[1])

# Load the array to be predicted
diabetes_data_to_predict = pd.read_csv('predicted.csv').to_numpy()

# Predict if person has diabetes
predictions = model.predict(diabetes_data_to_predict)

# Print predictions
for prediction in predictions:
    print('Person have got diabetes...' if prediction > .5 else 'Person is healthy...')

        

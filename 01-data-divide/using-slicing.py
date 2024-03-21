import pandas as pd
import numpy as np

# Define the divide ratio for splitting the dataset
DIVIDE_RATIO = .8

# Read diabetes data from csv file as a numpy array
diabetes_data = pd.read_csv('diabetes.csv').to_numpy()

# Shuffle dataset before dividing it
np.random.shuffle(diabetes_data)

# Divide dataset as train and test using slicing
training_size = int(np.round(len(diabetes_data) * DIVIDE_RATIO)) 

training_dataset = diabetes_data[:training_size, :]  # training section of the dataset
test_set = diabetes_data[training_size:, :]  # test section of the dataset

X_train = training_dataset[:, :-1]
y_train = training_dataset[:, -1]

X_test = test_set[:, :-1]
y_test = test_set[:, -1] 

import pandas as pd
from sklearn.model_selection import train_test_split

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

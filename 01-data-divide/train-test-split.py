import pandas as pd
from sklearn.model_selection import train_test_split

# Define the divide ratio for splitting the dataset
DIVIDE_RATIO = .8

# Read diabetes data from csv file as pandas DataFrame
diabetes_data = pd.read_csv('diabetes.csv')

# Split the dataset into training and test sets using train_test_split function
train_dataset, test_set = train_test_split(diabetes_data, test_size=1-DIVIDE_RATIO)

# Separate the input features (X_train) and output values (y_train) of the training dataset
X_train = train_dataset.iloc[:, :-1]
y_train = train_dataset.iloc[:, -1]

# Separate the input features (X_test) and output values (y_test) of the test dataset
X_test = test_set.iloc[:, :-1]
y_test = test_set.iloc[:, -1] 

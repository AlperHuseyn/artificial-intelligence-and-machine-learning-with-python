import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Set path for processed file to be saved in current working directory
file_name = 'titanic_R2U.csv'
PATH = f'{os.path.join(os.getcwd(), file_name)}'

# Read original titanic data from csv file
titanic_data = pd.read_csv('titanic.csv')
# print(f'All dataset with missing information: \n{titanic_data}', end='\n\n')

# Drop irrelevant columns from dataset
titanic_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
# print(f'Dataset after dropping "Name", "PassengerId", and "Ticket" columns: \n{titanic_data}', end='\n\n')

# Print column information for remaining columns in dataset
print(f'Column informations: \n{titanic_data.describe()}', end='\n\n')

# Print shape of dataset
print(f'Printing the shape of the dataset: {titanic_data.shape}', end='\n\n')

# Print column information for remaining columns in dataset
missing_counts = titanic_data.isna().sum().sum()
print(f'Missing data counts for the entire dataset: {missing_counts}', end='\n\n')

# Count number of missing values in each column
missing_counts_in_columns = titanic_data.isna().sum()
print(f'Missing data counts for each column: \n{missing_counts_in_columns}', end='\n\n')

# List column names with missing data
column_names_with_missing_data = [name for name in titanic_data.columns if titanic_data[name].isna().any()]
print('Columns with missing data:')
for column_name in column_names_with_missing_data:
    print(f'{column_name} -> type: {type(titanic_data[column_name][0])}')
print() # Add an extra empty line

# Calculate ratio of missing data across all columns
# isna() or isnull() can be used interchangeably, both for the same purpose
column_missing_data_ratio = titanic_data.isnull().sum().sum() / titanic_data.size
print(f'Ratio of missing data in all columns: \n{column_missing_data_ratio:.2f}')

# Count number of rows with missing data
rows_with_missing_data = titanic_data.isna().any(axis=1).sum()
print(f'Number of rows with missing data: \n{rows_with_missing_data}')

# Calculate ratio of missing data across all rows
row_missing_data_ratio = titanic_data.isna().any(axis=1).sum() / len(titanic_data)
print(f'Ratio of missing data in all rows: \n{row_missing_data_ratio:.2f}')

######################
# Imputation process #
######################

# Converting 'Sex' column using label encoding
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])

# Concerting 'Pclass' and 'Embarked' columns using one hot encoding
titanic_data = pd.get_dummies(titanic_data, columns=['Pclass', 'Embarked'], prefix=['', ''], prefix_sep='')

# Fill missing values in 'Age' column with median value
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing values in 'Fare' column with mean value
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)

# Checking there is no missing data left in the dataset
print(f'Last check for any remaining missing data: {titanic_data.isna().sum().sum()}')

# Save processed file into current working directory as a csv file
titanic_data.to_csv(PATH)

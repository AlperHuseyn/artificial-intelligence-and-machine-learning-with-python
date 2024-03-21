import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Set path for processed file to be saved in current working directory
file_name = 'bank_R2U.csv'
PATH = f'{os.path.join(os.getcwd(), file_name)}'

# Read original bank data from csv file
bank_data = pd.read_csv('bank.csv', delimiter=';')

# Replace 'unknown's with 'NaN'
bank_data.replace('unknown', np.nan, inplace=True)

# Print shape of dataset
print(f'Printing the shape of the dataset: {bank_data.shape}', end='\n\n')

# Print column information in dataset
print(f'Column informations: \n{bank_data.describe()}', end='\n\n')

# Missing info count for the entire dataset
missing_counts = bank_data.isna().sum().sum()
print(f'Missing data counts for the entire dataset: {missing_counts}', end='\n\n')

# Count number of missing values in each column
missing_counts_in_columns = bank_data.isna().sum()
print(f'Missing data counts for each column: \n{missing_counts_in_columns}', end='\n\n')

# List column names with missing data
column_names_with_missing_data = [name for name in bank_data.columns if bank_data[name].isna().any()]
print('Columns with missing data: ')
print(*column_names_with_missing_data, sep='\n', end='\n\n')

# The scale types of columns with missing data
print('Column scale types containing missing data:')
print(list(map(type, column_names_with_missing_data)), end='\n\n')

# Calculate ratio of missing data on 'job' column
job_column_missing_data_ratio = bank_data['job'].isnull().sum().sum() / len(bank_data)
print(f'Ratio of missing data in all columns: {job_column_missing_data_ratio:.4f}')

# Calculate ratio of missing data on 'education' column
education_column_missing_data_ratio = bank_data['education'].isnull().sum().sum() / len(bank_data)
print(f'Ratio of missing data in all columns: {education_column_missing_data_ratio:.4f}')

# Calculate ratio of missing data on 'contact' column
contact_column_missing_data_ratio = bank_data['contact'].isnull().sum().sum() / len(bank_data)
print(f'Ratio of missing data in all columns: {contact_column_missing_data_ratio:.4f}')

# Calculate ratio of missing data on 'poutcome' column
poutcome_column_missing_data_ratio = bank_data['poutcome'].isnull().sum().sum() / len(bank_data)
print(f'Ratio of missing data in all columns: {poutcome_column_missing_data_ratio:.4f}')

# Count number of rows with missing data
rows_with_missing_data = bank_data.isna().any(axis=1).sum()
print(f'Number of rows with missing data: {rows_with_missing_data}')

# Calculate ratio of missing data across all rows
row_missing_data_ratio = bank_data.isna().any(axis=1).sum() / len(bank_data)
print(f'Ratio of missing data in all rows: {row_missing_data_ratio:.2f}')

# Fill 'job', and 'education' columns with mode values
bank_data['job'].fillna(bank_data['job'].mode()[0], inplace=True)
bank_data['education'].fillna(bank_data['education'].mode()[0], inplace=True)

# Use one hot encoding on 'job', 'education', 'marital', and 'month' columns
bank_data = pd.get_dummies(bank_data, columns=['job', 'education', 'marital', 'month'], prefix=['', '', '', ''], prefix_sep='')

# Label encode related colums
label_encoder = LabelEncoder()
bank_data['default'] = label_encoder.fit_transform(bank_data['default'])
bank_data['loan'] = label_encoder.fit_transform(bank_data['loan'])
bank_data['housing'] = label_encoder.fit_transform(bank_data['housing'])
bank_data['y'] = label_encoder.fit_transform(bank_data['y'])

# Replace missing values with a default value 'unknown'
bank_data.fillna('unknown', inplace=True) 

# Convert 'contact' column to categorical and add unknown category
bank_data['contact'] = bank_data['contact'].astype('category')
bank_data['contact'] = bank_data['contact'].cat.add_categories('Unknown')
bank_data['contact'].replace('unknown', 'Unknown', inplace=True) 
bank_data['contact'] = pd.Categorical(bank_data['contact'], categories=['Unknown', 'telephone', 'cellular'])

# Use one hot encoding and replace 'unknown's on 'contact' column
bank_data = pd.get_dummies(bank_data, columns=['contact'], drop_first=True, prefix=[''], prefix_sep='')

# Convert 'poutcome' column to categorical and add unknown category
bank_data['poutcome'] = bank_data['poutcome'].astype('category')
bank_data['poutcome'] = bank_data['poutcome'].cat.add_categories('Unknown')
bank_data['poutcome'].replace('unknown', 'Unknown', inplace=True) 
bank_data['poutcome'] = pd.Categorical(bank_data['poutcome'], categories=['Unknown', 'success', 'failure', 'other'])

# Use one hot encoding and replace 'unknown's on 'poutcome' column
bank_data = pd.get_dummies(bank_data, columns=['poutcome'], drop_first=True, prefix=[''], prefix_sep='')

# Checking there is no missing data left in the dataset
print(f'Last check for any remaining missing data: {bank_data.isna().sum().sum()}')

# Save processed file into current working directory as a csv file
bank_data.to_csv(PATH)

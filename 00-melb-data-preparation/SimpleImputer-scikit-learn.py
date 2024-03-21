import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the Melbourne housing snapshot dataset
melb_data = pd.read_csv('melb_data.csv')

# Review missing data in the dataset
missing_counts = melb_data.isna().sum()
print('-----------------------------------------')
print(f'Missing data counts for each column: \n{missing_counts}')
print('-----------------------------------------')

columns_with_missing_data = [name for name in melb_data.columns if melb_data[name].isna().any()]
print('Columns with missing data:')
for column_name in columns_with_missing_data:
    print(f'{column_name}')
print('-----------------------------------------')

column_missing_data_ratio = melb_data.isna().sum().sum() / melb_data.size
print(f'Ratio of missing data in all columns: \n{column_missing_data_ratio:.2f}')
print('-----------------------------------------')

rows_with_missing_data = melb_data.isna().any(axis=1).sum()
print(f'Number of rows with missing data: \n{rows_with_missing_data}')
print('-----------------------------------------')

row_missing_data_ratio = melb_data.isna().any(axis=1).sum() / len(melb_data)
print(f'Ratio of missing data in all rows: \n{row_missing_data_ratio:.2f}')
print('-----------------------------------------')

######################
# Imputation process using skit-learn #
######################

# Create SimpleImputer object
melb_data_obj = SimpleImputer() # strategy:str, default = 'mean'
# Impute rounded mean into missing data for Car and BuildingArea columns
melb_data[['Car', 'BuildingArea']] = np.round(melb_data_obj.fit_transform(melb_data[['Car', 'BuildingArea']]))

# Impute mode into missing data for YearBuilt and CouncilArea columns
melb_data_obj.set_params(strategy='most_frequent')
melb_data[['YearBuilt', 'CouncilArea']] = melb_data_obj.fit_transform(melb_data[['YearBuilt', 'CouncilArea']])

# Checking there is no missing data left in the dataset
print(f'Last check for any remaining missing data: {melb_data.isna().sum().sum()}')
print('-----------------------------------------')

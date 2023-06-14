import pandas as pd
import numpy as np

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
# Imputation process #
######################

# Impute rounded mean into missing data for Car column
rounded_mean_of_Car_column = np.round(melb_data['Car'].mean())
melb_data['Car'].fillna(rounded_mean_of_Car_column, inplace=True)

# Impute rounded mean into missing data for BuildingArea column
rounded_mean_of_BuildingArea_column = np.round(melb_data['BuildingArea'].mean())
melb_data['BuildingArea'].fillna(rounded_mean_of_BuildingArea_column, inplace=True)

# Impute mode into missing data for YearBuilt column
melb_data['YearBuilt'].fillna(melb_data['YearBuilt'].mode()[0], inplace=True)

# Impute mode into missing data for CouncilArea column
melb_data['CouncilArea'].fillna(melb_data['CouncilArea'].mode()[0], inplace=True)

# Checking there is no missing data left in the dataset
print(f'Last check for any remaining missing data: {melb_data.isna().sum().sum()}')
print('-----------------------------------------')

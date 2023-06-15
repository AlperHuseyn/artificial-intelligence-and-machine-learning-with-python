----
# Download the Titanic Dataset

Download the Titanic dataset from the link below:

https://www.kaggle.com/datasets/brendan45774/test-file

## Dataset Appearance

The appearance of the dataset is as follows:

| PassengerId | Survived | Pclass | Name                                | Sex    | Age  | SibSp | Parch | Ticket   | Fare   | Cabin | Embarked |
| ----------- | -------- | ------ | ---------------------------------- | ------ | ---- | ----- | ----- | -------- | ------ | ------ | -------- |
| 892        | 0        | 3      | Kelly, Mr. James                     | male   | 34.5 | 0     | 0     | 330911   | 7.8292 |       | Q        |
| 893        | 1        | 3      | Wilkes, Mrs. James (Ellen Needs)    | female | 47   | 1     | 0     | 363272   | 7      |       | S        |
| 894        | 0        | 2      | Myles, Mr. Thomas Francis           | male   | 62   | 0     | 0     | 240276   | 9.6875 |       | Q        |
| 895        | 0        | 3      | Wirz, Mr. Albert                     | male   | 27   | 0     | 0     | 315154   | 8.6625 |       | S        |
| 896        | 1        | 3      | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female | 22   | 1     | 1     | 3101298  | 12.2875|       | S        |


# Prepare the Titanic Dataset

Use Pandas' `read_csv` function to read the dataset and perform the following preparation steps on the dataset:

## Data Preparation Steps
1. Drop the "Name", "PassengerId", and "Ticket" columns from the dataset.
2. Inspect the dataset for missing data and perform necessary "imputation" operations.
3. Convert the "Sex" column from categorical format of "female" and "male" to numerical format of 0 and 1.
4. Encode the "Pclass" column in "one hot encoding" format.
   - The "Pclass" column represents the passenger's class and takes values of 1, 2, 3.

## Considerations
- Categorical columns will be removed from the DataFrame object and the numerical columns will remain in the object.

- The dataset should be evaluated in terms of missing data and the following should be considered:
  - The number of rows and columns in the dataset
  - The ratio of missing data in rows and columns
  - The names of columns with missing data
  - The scale types of columns with missing data

- When performing imputation, the following techniques can be used - if appropriate:
  - The mean, median, or mode of the column
  - Treating missing data as a separate category

----

# Download the Bank Dataset

Download the Titanic dataset from the link below:

https://archive.ics.uci.edu/ml/datasets/bank+marketing

## Dataset Appearance

The appearance of the dataset is as follows:


| age |	job         | marital | education |	default | balance |	housing | loan | contact  | day | month | duration | campaign | pdays | previous | poutcome | y   |
| --- | ----------- | ------- | --------- | ------- | ------- | ------- | ---- | -------- | --- | ----- | -------- | -------- | ----- | -------- | -------- |-----|
| 30  |	unemployed  | married |	primary   |	no      | 1787    |	no      | no   | cellular |	19  | oct   | 79       | 1        |	-1    |	0        | unknown  | no  |
| 33  |	services    | married |	secondary |	no	    | 4789	  | yes     | yes  | cellular |	11  | may   | 220      | 1        |	339   |	4        | failure  | no  |
| 35  |	management  | single  |	tertiary  |	no	    | 1350	  | yes	    | no   | cellular | 16	| apr   | 185	   | 1	      | 330	  | 1	     | failure  | no  |
| 30  |	management  | married |	tertiary  |	no	    | 1476	  | yes     | yes  | unknown  |	3   | jun   | 199      | 4        |	-1    |	0        | unknown  | no  |
| 59  |	blue-collar | married |	secondary |	no      | 0       |	yes     | no   | unknown  |	5	| may   | 226      | 1	      | -1    |	0        | unknown  | no  |

# Prepare the bank.csv Dataset

1) Read the dataset using the `read_csv` function of Pandas. Although the dataset has a `'.csv'` extension, it uses `";"` as a separator.
So while reading the dataset using the read_csv function, use the argument `delimiter=';'`.

2) Perform the following operations on the dataset:

- Identify the categorical columns in the dataset and encode them as 0 and 1 if they contain two categories or as "one hot encoding"
if they contain more than two categories.

- There are some missing data in the columns in the form of `"unknown"`. If the missing data in the columns is more than 25%, express it
as a different category in the form of `"unknown"`. If the missing data in categorical columns is less than 25%, fill it using the "mode"
technique.

- Remove the categorical columns from the DataFrame object and add the numerically transformed columns to the object.

----

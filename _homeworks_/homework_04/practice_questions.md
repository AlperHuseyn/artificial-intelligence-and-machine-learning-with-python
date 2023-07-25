## Multi-Class Logistic Regression with Two Hidden Layer Artificial Neural Network

**Instructions:**

1. **Dataset Download:**
   - Download the dataset in XLS file format from the following link: [Cardiotocography Dataset](https://archive.ics.uci.edu/ml/datasets/cardiotocography). Save the file as "ctg.xls."

2. **Data Reading and Preprocessing:**
   - Use Pandas' `read_excel` function to read the Excel file.
   - Specify the worksheet name or index using the `sheet_name` parameter. For the "Raw Data" worksheet, the index is 2.
   - Select the columns of interest by setting `usecols='G:AN'` to read only columns from G to AN.
   - Remove rows with NaN values using the `dropna` function.
   - Choose the following columns from the DataFrame: `LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency, NSP`
   
```python
import pandas as pd

df_sheet = pd.read_excel('ctg.xls', sheet_name='Raw Data', usecols='G:AN')

df = df_sheet.dropna()
```

3. **Target Variable:**
   - The last column, NSP, represents the target categorical value to be predicted.
   - NSP can take values 1, 2, or 3, representing the patient's condition as follows:
     - NSP = 1 --> Normal patient
     - NSP = 2 --> Suspect patient
     - NSP = 3 --> Pathologic patient

4. **Model Building and Training:**
   - Create a two-hidden-layer artificial neural network for multi-class logistic regression.
   - Perform feature scaling as required.

5. **Model Testing and Prediction:**
   - Test the trained model with the test dataset.
   - Make a sample prediction using the model.

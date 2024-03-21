"""
This script loads a CSV file containing diabetes data to be predicted and uses a pre-trained model to predict whether
each person in the dataset has diabetes or not. The predictions are printed to the console.
"""


from tensorflow.keras.models import load_model
import pandas as pd

model = load_model('keras-predictor.h5')

# Load the array to be predicted
diabetes_data_to_predict = pd.read_csv('predicted.csv').to_numpy()

# Predict if person has diabetes
predictions = model.predict(diabetes_data_to_predict)

# Print predictions
for prediction in predictions:
    print('Person have got diabetes...' if prediction > .5 else 'Person is healthy...')

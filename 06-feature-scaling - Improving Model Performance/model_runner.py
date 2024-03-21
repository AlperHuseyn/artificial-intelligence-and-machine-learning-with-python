from tensorflow.keras.models import load_model
import pickle
import pandas as pd


model = load_model('heart-failure.h5')

with open('heart-failure.pickle', 'rb') as f:
    scaler = pickle.load(f)
    
# Load the array to be predicted and perform feature scaling on it
heart_failure_data_to_predict = pd.read_csv('predicted.csv').to_numpy()
feature_scaled_to_predict = scaler.transform(heart_failure_data_to_predict)

predictions = model.predict(feature_scaled_to_predict)
# Print the predicted risk of heart failure for each individual in the array
for prediction in predictions:
    print('*heart failure risk is high...' if prediction > .5 else 'Person has a low risk of heart failure...')
    
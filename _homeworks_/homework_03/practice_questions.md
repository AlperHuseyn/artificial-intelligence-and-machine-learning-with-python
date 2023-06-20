## Heart Failure Clinical Records Dataset Analysis and Prediction

We will analyze the `"heart_failure_clinical_records_dataset.csv"` dataset that was created to predict whether a person will die from heart problems based on various biomedical measurements. You can download this dataset from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data).

Since there are significant scale differences between columns in the dataset, feature scaling should be applied for better performance. However, we will not apply it in this case.

To start with, we will split the dataset into 90% training and 10% test datasets. Then, we will create an artificial neural network model using Keras for a two-class classification model with two hidden layers. The number of neurons in each layer can be decided on later.

We will train the model with any optimizer algorithm, selecting appropriate loss functions and metric values. We will use the validation set, consisting of 10% of the training data, for training and check the training and validation metrics during training.

Next, we will repeat the training with epoch values of 1, 50, 100, 150, and 200. Finally, we will test the final models with the test dataset.

To demonstrate the predictive power of our model, we will make predictions by taking some data from the test dataset and changing some column values.

# Admission_data_regression
My first semi-autonomous Deep Learning project

This project is a deep learning regression model to determine the acceptance probability for students applying to graduate school. The model is built using Keras with TensorFlow.

Data is taken from the Graduation Admission2 dataset locate here: https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv

Seeing as only 500 data points are available, a 5 ply K-fold cross-validation was used to properly validate data. The neural network currently has a single hidden layer containing 64 neurons. This layer uses ReLu asits activation function. The Adam optimizer is used for the gradiant descent. Early stopping is integrated into the model.

## General model parameters are:

### Model parameters
num_epochs = 400  
batch_size = 16  
nb_neurons = 64  
learning_rate = 0.001  
### EarlyStopping
min_delta = 0.00001  
patience = 20  
### Because we don't have a lot of data, we should use k-fold validation
k = 5  # Number of units  

## Conclusion
While the model does show signs of overfitting, it still currently has an accuracy over 80%.

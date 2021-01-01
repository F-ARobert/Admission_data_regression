import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Import data
dataset = pd.read_csv("admissions_data.csv")
# print(admission_data.head())

# Select features and labels
# Serial No. (col 0) is not to be included in features.
dataset = dataset.drop(['Serial No.'], axis=1)  # removes serial numbers from the data
# print(admission_data.head())

# select labels
labels = dataset.iloc[:-1]

# Features are from col. 1 to 7
features = dataset.iloc[:, 0:-1]  # Select all rows from all columns save last
#print(features.head())
#print(labels.describe())

# Split training data and test data
# Because we don't have a lot of data, we should use k-fold validation
# Todo implement k-fold data validation.

# Number of k units
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=15)

# Normalize data
# List all numerical features. This allows to select numerical features (float64 or int64) automatically
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

# Creating the transformer that will be applied to the data
ct = ColumnTransformer([("only numeric", Normalizer(), numerical_columns)], remainder='passthrough')


# Create regression model
model = Sequential()

# Create input layer to the network model
inputs = InputLayer(activation='relu', input_shape=(features.shape[1],))

# Add layers to model
model.add(inputs)
model.add(Dense(32, activation="relu"))
model.add(Dense(32))
model.add(Dense(1)) # Model output. Regression model == single output

print(model.summary())

########################################################################
# Initialize optimizer and compile model
########################################################################
# Create instance of Adam
opt = Adam(learning_rate=0.01)

# Compile model
model.compile(loss='mse', metrics=['mae'], optimizer=opt)

for train_index, test_index in kf.split(features):
    pass
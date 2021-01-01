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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def create_model():
    # Create regression model
    base_model = Sequential()

    # Create input layer to the network model
    inputs = InputLayer(activation='relu', input_shape=(features.shape[1],))

    # Add layers to model
    base_model.add(inputs)
    base_model.add(Dense(32, activation="relu"))
    base_model.add(Dense(32))
    base_model.add(Dense(1))  # Model output. Regression model == single output

    # Initialize optimizer and compile model
    # Create instance of Adam
    opt = Adam(learning_rate=0.01)

    # Compile model
    base_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

    return base_model


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
# print(features.head())
# print(labels.describe())

# Split training data and test data
# Because we don't have a lot of data, we should use k-fold validation
# Number of k units
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=15)

# Normalize data
# List all numerical features. This allows to select numerical features (float64 or int64) automatically
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

# Creating the transformer that will be applied to the data
ct = ColumnTransformer([("only numeric", Normalizer(), numerical_columns)], remainder='passthrough')

model = create_model()

print(model.summary())


mse_score = []  # List to hold individual performances
mae_score = []  # List to hold individual performances

for train_index, test_index in kf.split(features):
    features_train, features_test = features.iloc[train_index, :], features.iloc[test_index, :]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Apply fit column transformer to training data
    features_train_scaled = ct.fit_transform(features_train)

    # Transform test data using ct columntransformer instance
    features_test_scaled = ct.transform(features_test)

    model.fit(features_train, labels_train, epochs=100, batch_size=8, verbose=1)
    pred_values = model.predict(features_test)

    acc = accuracy_score(pred_values, labels_test)
    acc_score.append(acc)

avg_acc_score = sum(acc_score) / k

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

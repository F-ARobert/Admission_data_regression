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
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold

num_epochs = 154
batch_size = 8

def create_model():
    # Create regression model
    base_model = Sequential()

    # Create input layer to the network model
    inputs = InputLayer(input_shape=(features.shape[1],))

    # Add layers to model
    base_model.add(inputs)
    base_model.add(Dense(16, activation="relu"))
    base_model.add(Dense(32, activation="relu"))
    base_model.add(Dense(1))  # Model output. Regression model == single output

    # Initialize optimizer and compile model
    # Create instance of Adam
    opt = Adam(learning_rate=0.01)

    # Compile model
    base_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

    return base_model


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 -
                                                                factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


# Import data
dataset = pd.read_csv("admissions_data.csv")
# print(admission_data.head())

# Select features and labels
# Serial No. (col 0) is not to be included in features.
dataset = dataset.drop(['Serial No.'], axis=1)  # removes serial numbers from the data
# print(admission_data.head())

# select labels
labels = dataset.iloc[:, -1]

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

# Lists to hold individual performances
mse_score = []
mae_score = []
acc_score = []
all_mae_histories = []
all_val_mae_histories = []

for train_index, test_index in kf.split(features):
    features_train, features_test = features.iloc[train_index, :], features.iloc[test_index, :]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Apply fit column transformer to training data
    features_train_scaled = ct.fit_transform(features_train)

    # Transform test data using ct columntransformer instance
    features_test_scaled = ct.transform(features_test)

    train_history = model.fit(features_train_scaled, labels_train, validation_data=(features_test_scaled, labels_test),
                              epochs=num_epochs, batch_size=batch_size, verbose=1)

    # Create list of MAE history
    mae_history = train_history.history['mae']
    val_mae_history = train_history.history['val_mae']
    all_mae_histories.append(mae_history)
    all_val_mae_histories.append(val_mae_history)

    # Evaluate model
    res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)
    mae_score.append(res_mae)
    mse_score.append(res_mae)

    # Evaluate accuracy of predictions
    pred_values = model.predict(features_test_scaled)
    acc = r2_score(labels_test, pred_values)
    acc_score.append(acc)

# Calculate MAE average
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_val_mae_history = [np.mean([x[i] for x in all_val_mae_histories]) for i in range(num_epochs)]

# Calculate average accuracy score
avg_acc_score = sum(acc_score) / k

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

# Show plot of MAE vs epochs in training
smooth_mae_history = smooth_curve(average_mae_history)
smooth_val_mae_history = smooth_curve(average_val_mae_history)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(smooth_mae_history)
ax1.plot(smooth_val_mae_history)
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
plt.show()

# todo Implement mat plot lib plots of error vs epochs
# todo Add Early stopping to prevent overfitting

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold

num_epochs = 50
batch_size = 16


def create_model():
    # Create regression model
    base_model = Sequential()

    # Create input layer to the network model
    inputs = InputLayer(input_shape=(features.shape[1],))

    # Add layers to model
    base_model.add(inputs)
    base_model.add(Dense(8, activation="relu"))
    base_model.add(Dense(8, activation="relu"))
    base_model.add(Dense(1))  # Model output. Regression model == single output

    # Initialize optimizer and compile model
    # Create instance of Adam
    opt = Adam(learning_rate=0.005)

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


def find_max_length(array):
    max_length = 0
    for i in array:
        if len(i) > max_length:
            max_length = len(i)

    return max_length


def pad_array(array):
    max_length = find_max_length(array)
    array_pad = []
    for i in array:
        len(i) < max_length
        diff = max_length - len(i)
        array_pad.append(np.pad(i, (0, diff), 'constant', constant_values=i[-1]))

    return array_pad, max_length

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
# Creating the transformer that will be applied to the data
norm = Normalizer()

# Create callback for EarlyStopping
# Min_delta represents 0.01%
callback = EarlyStopping(monitor='loss', min_delta=0.0001, patience=20, mode='min', restore_best_weights=True)

# Create model
model = create_model()

# print(model.summary())

# Lists to hold individual performances
loss = []
val_loss = []
acc_score = []
all_mae_histories = []
all_val_mae_histories = []

# For loop permitting testing and validating over all 5 data sub-units
for train_index, test_index in kf.split(features):
    features_train, features_test = features.iloc[train_index, :], features.iloc[test_index, :]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Apply normalization to training data
    features_train_scaled = norm.fit_transform(features_train)
    features_test_scaled = norm.transform(features_test)

    # Uncomment to visualize normalized data
    # features_train_scaled = pd.DataFrame(features_train_scaled, columns=features_train.columns)
    # features_test_scaled = pd.DataFrame(features_test_scaled, columns=features_test.columns)
    # print(features_train_scaled.describe())
    # print(features_test_scaled.describe())

    train_history = model.fit(features_train_scaled, labels_train, validation_data=(features_test_scaled, labels_test),
                              epochs=num_epochs, batch_size=batch_size, verbose=1, callbacks=[callback])

    # print(train_history.history.keys())

    # Create list of MAE history
    loss.append(train_history.history['loss'])
    val_loss.append(train_history.history['val_loss'])
    all_mae_histories.append(train_history.history['mae'])
    all_val_mae_histories.append(train_history.history['val_mae'])

    # Evaluate model
    # res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)
    # mae_score.append(res_mae)
    # mse_score.append(res_mae)

    # Evaluate accuracy of predictions
    pred_values = model.predict(features_test_scaled)
    acc = r2_score(labels_test, pred_values)
    acc_score.append(acc)


# Before averages are calculated, reshape vector lengths
loss_pad, max_epoch = pad_array(loss)
val_loss_pad, max_epoch = pad_array(val_loss)
all_mae_histories_pad, max_epoch = pad_array(all_mae_histories)
all_val_mae_histories_pad, max_epoch = pad_array(all_val_mae_histories)

# Calculate MAE and loss averages
average_mae_history = [np.mean([x[i] for x in all_mae_histories_pad]) for i in range(max_epoch)]
average_val_mae_history = [np.mean([x[i] for x in all_val_mae_histories_pad]) for i in range(max_epoch)]
average_loss = [np.mean([x[i] for x in loss_pad]) for i in range(max_epoch)]
average_val_loss = [np.mean([x[i] for x in val_loss_pad]) for i in range(max_epoch)]

# Calculate average accuracy score
avg_acc_score = np.mean(acc_score)

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

# Smooth lines for plots of MAE and loss vs epochs in training
smooth_mae_history = smooth_curve(average_mae_history)
smooth_val_mae_history = smooth_curve(average_val_mae_history)
smooth_loss = smooth_curve(average_loss)
smooth_val_loss = smooth_curve(average_val_loss)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(smooth_mae_history)
ax1.plot(smooth_val_mae_history)
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper right')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(smooth_loss)
ax2.plot(smooth_val_loss)
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper right')
plt.tight_layout()
plt.show()

# todo Implement mat plot lib plots of error vs epochs
# todo Add Early stopping to prevent overfitting

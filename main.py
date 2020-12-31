import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

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




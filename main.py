# Importing the Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Get the Dataset
df = pd.read_csv("NXE.csv", na_values=['null'], index_col='Date',parse_dates=True, infer_datetime_format=True)
df.head()
print("Dataframe Shape: ", df.shape)
df['Adj Close'].plot()

plt.show()

# Split training and testing data for time-series
data = pd.read_csv("NXE.csv", na_values=['null'], parse_dates=True, infer_datetime_format=True)
data.head()
data_length = len(data)
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(data))
test_size = int(test_ratio * len(data))
print("train_size: " + str(train_size))
print("test_size: " + str(test_size))

train = data[:train_size][['Date', 'Close']]
test = data[train_size:][['Date', 'Close']]
print(train)

X, y = [], []

for i in range(train_size, data_length):
    X.append(data[i-data_length:i])
    # y.append(data[i])

print(np.array(X))
# print(np.array(y))

training_set = train.iloc[:, 1:2].values
print(training_set)
print(training_set.shape)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_training_set = scaler.fit_transform(training_set)
print(scaled_training_set)

X_train = []
y_train = []
for i in range(test_size, train_size):
    X_train.append(scaled_training_set[i-test_size:i, 0])
    y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

dataset_test = pd.read_csv("NXE.csv")
actual_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((data['Open'], dataset_test['Open']))

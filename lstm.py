import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from historicalVolData import historical_volatility  # Make sure this returns a DataFrame

# Parameters
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = 'AAPL'
WINDOW_DAYS = 100  # Rolling window for calculating volatility
SEQUENCE_LENGTH = 30  # Number of previous days the LSTM will look at

# Step 1: Load Data and Calculate Volatility
data = historical_volatility(TICKER, START_DATE, END_DATE)

# Drop NaNs resulting from rolling calculations
data.dropna(inplace=True)

# Step 2: Prepare Data for LSTM Regression
# We'll use TimeseriesGenerator to create sequences of the last SEQUENCE_LENGTH days for each input
features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
target = data['Volatility'].values  # Volatility as the continuous target for regression

# Split into training and testing data
train_size = int(0.7 * len(features))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Create time series sequences for training and testing
train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, batch_size=32)
test_gen = TimeseriesGenerator(X_test, y_test, length=SEQUENCE_LENGTH, batch_size=32)

# Step 3: Define and Compile LSTM Model for Regression
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train.shape[1])))
model.add(Dense(1))  # Single output neuron with linear activation for regression

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # MSE loss for regression

# Step 4: Train the Model
history = model.fit(train_gen, epochs=20, validation_data=test_gen)

# Step 5: Evaluate the Model
# Predictions for test set
y_pred = model.predict(test_gen)

# Align the lengths for evaluation (drop initial SEQUENCE_LENGTH observations)
y_test_flat = y_test[SEQUENCE_LENGTH:]
y_pred_flat = y_pred.flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test_flat, y_pred_flat)
mae = mean_absolute_error(y_test_flat, y_pred_flat)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from data.historicalVolData import historical_volatility  # Make sure this returns a DataFrame
#
# # Parameters
# START_DATE = '2013-01-01'
# END_DATE = '2023-12-31'
# TICKER = 'AAPL'
# WINDOW_DAYS = 100  # Rolling window for calculating volatility
# SEQUENCE_LENGTH = 30  # Number of previous days the LSTM will look at
#
# # Step 1: Load Data and Calculate Volatility
# data = historical_volatility(TICKER, START_DATE, END_DATE)
# print(data)
# # Drop NaNs resulting from rolling calculations
# data.dropna(inplace=True)
#
# # Step 2: Prepare Data for LSTM Regression
# # We'll use TimeseriesGenerator to create sequences of the last SEQUENCE_LENGTH days for each input
# features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
# target = data['Volatility'].values  # Volatility as the continuous target for regression
#
# # Split into training and testing data
# train_size = int(0.7 * len(features))
# X_train, X_test = features[:train_size], features[train_size:]
# y_train, y_test = target[:train_size], target[train_size:]
#
# # Create time series sequences for training and testing
# train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, batch_size=32)
# test_gen = TimeseriesGenerator(X_test, y_test, length=SEQUENCE_LENGTH, batch_size=32)
#
# # Step 3: Define and Compile LSTM Model for Regression
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train.shape[1])))
# model.add(Dense(1))  # Single output neuron with linear activation for regression
#
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # MSE loss for regression
# print("model compiled")
# # Step 4: Train the Model
# history = model.fit(train_gen, epochs=20, validation_data=test_gen)
# print("model fit")
# # Step 5: Evaluate the Model
# # Predictions for test set
# y_pred = model.predict(test_gen)
#
# # Align the lengths for evaluation (drop initial SEQUENCE_LENGTH observations)
# y_test_flat = y_test[SEQUENCE_LENGTH:]
# y_pred_flat = y_pred.flatten()
#
# # Calculate performance metrics
# mse = mean_squared_error(y_test_flat, y_pred_flat)
# mae = mean_absolute_error(y_test_flat, y_pred_flat)
#
# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.historicalVolData import generate_ticker_vol

# Parameters
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = 'AAPL'
WINDOW_DAYS = 100
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
# Step 1: Load Data and Calculate Volatility
data = generate_ticker_vol(TICKER, START_DATE, END_DATE)
data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

# Prepare features and target
features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
target = data['Volatility'].values  # Volatility as the continuous target

# Step 2: Prepare Data for LSTM Regression
# Split into training and testing data
train_size = int(0.7 * len(features))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create sequences for LSTM input
def create_sequences(features, targets, sequence_length):
    sequences, labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        labels.append(targets[i + sequence_length])
    return torch.stack(sequences), torch.tensor(labels)


X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, SEQUENCE_LENGTH)

# Create data loaders
train_data = TensorDataset(X_train_seq, y_train_seq)
test_data = TensorDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# Step 3: Define the LSTM Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


model = LSTMRegressor(input_dim=1, hidden_dim=50, output_dim=1).to(DEVICE)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader)}")

# Step 5: Evaluate the Model
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        predictions = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(y_batch.numpy())

# Align the lengths for evaluation (drop initial SEQUENCE_LENGTH observations)
y_test_flat = y_test_seq.numpy()
y_pred_flat = np.array(y_pred)

# Calculate performance metrics
mse = mean_squared_error(y_test_flat, y_pred_flat)
mae = mean_absolute_error(y_test_flat, y_pred_flat)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

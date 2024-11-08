"""
run_baseline.py
-------------------
Run baseline LSTM model with date-aware plotting
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import vol_loader
from models.lstm import LSTMRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# consts
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###################### functionality

# 1) Load data, prepare features and target
data = vol_loader.load_ticker_data(TICKER)

data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

# Check if 'Date' column exists; if not, use the index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date'].values
else:
    dates = data.index.values

features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
target = data['Volatility'].values  # Volatility as the continuous target

# 2) train-test split including dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    features, target, dates, train_size=SPLIT_SIZE, shuffle=False
)

# 3) Normalize the data
# Initialize scalers
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit scalers on training data
feature_scaler.fit(X_train)
target_scaler.fit(y_train.reshape(-1, 1))

# Transform features
X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Transform targets
y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Convert scaled data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 4) Create sequences for LSTM input, including dates
def create_sequences(features, targets, dates, sequence_length):
    sequences, labels, seq_dates = [], [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        labels.append(targets[i + sequence_length])
        seq_dates.append(dates[i + sequence_length])
    return torch.stack(sequences), torch.tensor(labels), seq_dates


X_train_seq, y_train_seq, dates_train_seq = create_sequences(
    X_train_tensor, y_train_tensor, dates_train, SEQUENCE_LENGTH
)
X_test_seq, y_test_seq, dates_test_seq = create_sequences(
    X_test_tensor, y_test_tensor, dates_test, SEQUENCE_LENGTH
)

# 5) Data Loaders
train_data = TensorDataset(X_train_seq, y_train_seq)
test_data = TensorDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 6) Initialize Model, Criterion, Optimizer
model = LSTMRegressor(input_dim=1, hidden_dim=50, output_dim=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Optional: Initialize weights (recommended for better performance)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

model.apply(init_weights)

# 7: Train the Model
train_losses = []  # To store training loss for each epoch

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        # Move to device
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss}")

# Optional: Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# 8: Evaluate the Model
model.eval()
y_pred = []
y_true = []
pred_dates = []  # To store dates corresponding to predictions

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        predictions = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(y_batch.numpy())

# Inverse transform the predictions and true values
y_pred_inverse = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
y_true_inverse = target_scaler.inverse_transform(y_test_seq.numpy().reshape(-1, 1)).flatten()

# Align dates with the predictions
pred_dates = dates_test_seq

# Ensure that pred_dates has the same length as y_pred_inverse and y_true_inverse
assert len(pred_dates) == len(y_pred_inverse) == len(y_true_inverse), \
    "Mismatch in lengths of dates and prediction arrays."

# Calculate performance metrics
mse = mean_squared_error(y_true_inverse, y_pred_inverse)
mae = mean_absolute_error(y_true_inverse, y_pred_inverse)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

print(pred_dates)
# 9: Plot Predictions vs Actual with Dates
def plot_preds(dates, y_true, y_pred):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label="Actual Volatility", color='blue')
    plt.plot(dates, y_pred, label="Predicted Volatility", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("LSTM Model - Predicted vs. Actual Volatility")
    plt.legend()

    # Improve date formatting on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    plt.tight_layout()
    plt.show()

plot_preds(pred_dates, y_true_inverse, y_pred_inverse)

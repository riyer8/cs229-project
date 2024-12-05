"""
run_gru_baseline.py
-------------------
Run baseline GRU model with date-aware plotting
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# consts
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###################### functionality

# Load data, prepare features and target
data = vol_loader.load_ticker_data(TICKER)
data.dropna(inplace=True)

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date'].values
else:
    dates = data.index.values

features = data[['Daily Return']].values
target = data['Volatility'].values

# Train-test split including dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    features, target, dates, train_size=SPLIT_SIZE, shuffle=False
)

# Normalize the data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(X_train)
target_scaler.fit(y_train.reshape(-1, 1))

X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Convert scaled data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create sequences for GRU input
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

train_data = TensorDataset(X_train_seq, y_train_seq)
test_data = TensorDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Define GRU model
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

model = GRURegressor(input_dim=1, hidden_dim=50, output_dim=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Evaluate the model
model.eval()
y_pred = []
y_true = []
pred_dates = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        predictions = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(y_batch.numpy())

y_pred_inverse = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
y_true_inverse = target_scaler.inverse_transform(y_test_seq.numpy().reshape(-1, 1)).flatten()
pred_dates = dates_test_seq

mse = mean_squared_error(y_true_inverse, y_pred_inverse)
mae = mean_absolute_error(y_true_inverse, y_pred_inverse)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Plot Predictions vs Actual with Dates
def plot_preds(dates, y_true, y_pred):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label="Actual Volatility", color='blue')
    plt.plot(dates, y_pred, label="Predicted Volatility", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("GRU Model - Predicted vs. Actual Volatility")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

plot_preds(pred_dates, y_true_inverse, y_pred_inverse)

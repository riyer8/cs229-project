"""
log transformed features
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
import seaborn as sns

# ----------------------------
# Constants and Hyperparameters
# ----------------------------
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# 1) Load Data and Prepare Features
# ----------------------------
# Load the data using your custom loader
data = vol_loader.load_ticker_data(TICKER)
data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

# ----------------------------
# 2) Handle 'Date' Column
# ----------------------------
# Convert 'Date' column to datetime if it exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date']
else:
    # If 'Date' is not a column, create it from the index
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date']

# Display first few rows to understand the data structure
print("First few rows of the data:")
print(data.head())

# ----------------------------
# 3) Select Features and Target
# ----------------------------
# Select the relevant feature columns and the target variable
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return']].copy()
target = data['Volatility'].copy()  # Continuous target

# ----------------------------
# 4) Ensure All Feature Columns Are Numeric
# ----------------------------
# Convert all feature columns to numeric types, coercing errors to NaN
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

# Check for any NaNs introduced by conversion
print("\nNumber of NaNs in each feature after conversion:")
print(features.isnull().sum())

# Combine features, target, and dates into a single DataFrame for synchronized dropping
combined_df = features.copy()
combined_df['Volatility'] = target
combined_df['Date'] = dates

# Drop rows with any NaNs in features or target
initial_length = len(combined_df)
combined_df.dropna(inplace=True)
final_length = len(combined_df)
print(f"\nDropped {initial_length - final_length} rows due to non-numeric data.")

# Extract cleaned features, target, and dates
features_clean = combined_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return']]
target_clean = combined_df['Volatility'].values
dates_clean = combined_df['Date'].values

# ----------------------------
# 5) Data Transformation (Log Transformation for Skewed Features)
# ----------------------------
# Identify skewed features
skewed_features = ['Volume']  # Add other skewed features if necessary

for feature in skewed_features:
    if feature in features_clean.columns:
        # Check for non-positive values before log transformation
        if (features_clean[feature] <= 0).any():
            print(f"Feature '{feature}' contains non-positive values. Adding 1 before log transformation.")
            features_clean[feature] = np.log1p(features_clean[feature] + 1)
        else:
            features_clean[feature] = np.log1p(features_clean[feature])
    else:
        print(f"Feature '{feature}' not found in the DataFrame.")

# Plot distribution after transformation
plt.figure(figsize=(10, 6))
for feature in skewed_features:
    if feature in features_clean.columns:
        sns.histplot(features_clean[feature], kde=True, label=f'Log Transformed {feature}')
plt.legend()
plt.title('Distribution of Log-Transformed Features')
plt.show()

# ----------------------------
# 6) Train-Test Split Including Dates
# ----------------------------
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    features_clean, target_clean, dates_clean, train_size=SPLIT_SIZE, shuffle=False
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ----------------------------
# 7) Normalize the Data
# ----------------------------
# Initialize scalers
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit scalers on training data
feature_scaler.fit(X_train)
target_scaler.fit(y_train.reshape(-1, 1))

# Transform features and targets
X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Debug: Check scaling statistics
print("\nFeature Scaling Statistics:")
print(f"Mean of scaled features: {X_train_scaled.mean(axis=0)}")
print(f"Std Dev of scaled features: {X_train_scaled.std(axis=0)}")

print("\nTarget Scaling Statistics:")
print(f"Mean of scaled target: {y_train_scaled.mean()}")
print(f"Std Dev of scaled target: {y_train_scaled.std()}")

# Plot histograms of scaled features
scaled_df = pd.DataFrame(X_train_scaled, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return'])
plt.figure(figsize=(15, 10))
for idx, column in enumerate(scaled_df.columns):
    plt.subplot(3, 2, idx+1)
    sns.histplot(scaled_df[column], kde=True)
    plt.title(f'Distribution of Scaled {column}')
plt.tight_layout()
plt.show()

# ----------------------------
# 8) Convert Scaled Data to Tensors
# ----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# ----------------------------
# 9) Create Sequences for LSTM Input
# ----------------------------
def create_sequences(features, targets, dates, sequence_length):
    sequences, labels, seq_dates = [], [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        labels.append(targets[i + sequence_length])
        seq_dates.append(dates[i + sequence_length])
    return torch.stack(sequences), torch.tensor(labels), seq_dates

# Create sequences for training and testing
X_train_seq, y_train_seq, dates_train_seq = create_sequences(
    X_train_tensor, y_train_tensor, dates_train, SEQUENCE_LENGTH
)
X_test_seq, y_test_seq, dates_test_seq = create_sequences(
    X_test_tensor, y_test_tensor, dates_test, SEQUENCE_LENGTH
)

print(f"\nTraining sequences: {X_train_seq.shape}, {y_train_seq.shape}")
print(f"Testing sequences: {X_test_seq.shape}, {y_test_seq.shape}")

# ----------------------------
# 10) Create Data Loaders
# ----------------------------
train_data = TensorDataset(X_train_seq, y_train_seq)
test_data = TensorDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 11) Initialize Model, Criterion, Optimizer
# ----------------------------
input_dim = features_clean.shape[1]  # Number of feature columns
output_dim = 1  # Predicting a single value: Volatility

# Initialize the LSTM model (ensure no 'num_layers' or 'dropout' are passed)
model = LSTMRegressor(input_dim=input_dim, hidden_dim=50, output_dim=output_dim).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel Initialized:")
print(model)

# ----------------------------
# 12) Optional: Initialize Weights (Xavier Initialization)
# ----------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

# Apply weight initialization
model.apply(init_weights)

# ----------------------------
# 13) Training the Model
# ----------------------------
train_losses = []  # To store training loss for each epoch

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.6f}")

# ----------------------------
# 14) Plot Training Loss
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# ----------------------------
# 15) Evaluate the Model on Test Set
# ----------------------------
model.eval()
y_pred = []
y_true = []
pred_dates = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        predictions = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(y_batch.cpu().numpy())

# Inverse transform the predictions and true values
y_pred_inverse = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
y_true_inverse = target_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()

pred_dates = dates_test_seq  # Corresponding dates

# Ensure alignment
assert len(pred_dates) == len(y_pred_inverse) == len(y_true_inverse), \
    "Mismatch in lengths of dates and prediction arrays."

# Calculate performance metrics
mse = mean_squared_error(y_true_inverse, y_pred_inverse)
mae = mean_absolute_error(y_true_inverse, y_pred_inverse)

print(f"\nTest Mean Squared Error (MSE): {mse:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

# ----------------------------
# 16) Plot Predictions vs Actual Values
# ----------------------------
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

# Plot the predictions vs actual values
plot_preds(pred_dates, y_true_inverse, y_pred_inverse)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HORIZON = 14

class LSTMVolatilityPredictor:
    def __init__(self, sequence_length=30, batch_size=16, epochs=20, hidden_dim=50, learning_rate=0.001, device=None):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.train_losses = []

    def prepare_data(self, data, features, target, split_size=0.7, val_size=0.2):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            dates = data['Date'].values
        else:
            dates = data.index.values

        X = data[features].values
        y = data[target].values

        train_end = int(len(X) * split_size)
        val_end = train_end + int(len(X) * val_size)

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        dates_train, dates_val, dates_test = dates[:train_end], dates[train_end:val_end], dates[val_end:]

        # Scale features and targets
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train.reshape(-1, 1))

        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        y_train_scaled = self.target_scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        def create_sequences(features, targets, dates, sequence_length, horizon=HORIZON):
            sequences, labels, seq_dates = [], [], []
            max_index = len(features) - sequence_length - horizon
            for i in range(max_index):
                seq = features[i:i + sequence_length]
                target = targets[i + sequence_length + horizon - 1]
                date = dates[i + sequence_length + horizon - 1]
                sequences.append(seq)
                labels.append(target)
                seq_dates.append(date)
            return torch.stack(sequences), torch.tensor(labels), seq_dates

        # create sequences
        X_train_seq, y_train_seq, dates_train_seq = create_sequences(X_train_tensor, y_train_tensor, dates_train,
                                                                     self.sequence_length)
        X_val_seq, y_val_seq, dates_val_seq = create_sequences(X_val_tensor, y_val_tensor, dates_val,
                                                               self.sequence_length)
        X_test_seq, y_test_seq, dates_test_seq = create_sequences(X_test_tensor, y_test_tensor, dates_test,
                                                                  self.sequence_length)

        # Data loaders
        train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, dates_val_seq, y_val_seq, dates_test_seq, y_test_seq

    def build_model(self, input_dim, output_dim=1):
        class LSTMRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(LSTMRegressor, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward_alternative(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM
                return out

            def forward(self, x):
                _, (hn, _) = self.lstm(x)
                out = self.fc(hn[-1])
                return out

        self.model = LSTMRegressor(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim).to(
            self.device)

    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}")

    def evaluate(self, test_loader, y_test_seq, dates_test_seq):
        self.model.eval()
        y_pred = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch).squeeze().cpu().numpy()
                y_pred.extend(predictions)

        # Inverse transform predictions and true values
        y_pred_inverse = self.target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
        y_true_inverse = self.target_scaler.inverse_transform(y_test_seq.numpy().reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_true_inverse, y_pred_inverse)
        mae = mean_absolute_error(y_true_inverse, y_pred_inverse)
        r2 = r2_score(y_true_inverse, y_pred_inverse)
        correlation_coefficient = np.corrcoef(y_true_inverse, y_pred_inverse)[0, 1]
        explained_variance = explained_variance_score(y_true_inverse, y_pred_inverse)
        metrics = {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "R-squared": r2,
            "Correlation Coefficient": correlation_coefficient,
            "Explained Variance": explained_variance
        }

        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return dates_test_seq, y_true_inverse, y_pred_inverse, metrics

    def plot_loss(self, save_path="lstm_loss.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('LSTM Training Loss on Dow Jones Data Over Epochs')
        plt.legend()
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Training loss plot saved to {save_path}")
        plt.show()

    def plot_predictions(self, dates, y_true, y_pred, horizon=HORIZON, save_path="lstm_preds.png"):
        plt.figure(figsize=(15, 7))
        plt.plot(dates, y_true, label="Actual Volatility", color='blue')
        plt.plot(dates, y_pred, label="Predicted Volatility", color='orange')
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.title(f"LSTM - Predicted vs. Actual Volatility on Dow Jones Test Data (Horizon: {horizon} Days)")
        plt.legend()

        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to {save_path}")
        plt.show()
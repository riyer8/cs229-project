"""
Define LSTM Model
"""
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

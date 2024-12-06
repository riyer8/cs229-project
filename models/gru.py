import torch.nn as nn

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
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import vol_loader
import matplotlib.pyplot as plt


SPLIT_SIZE = 0.7
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20

# Load your data
data = vol_loader.load_ticker_data(TICKER)


"""
# Load your data
data = vol_loader.load_ticker_data(TICKER)

# Ensure data is sorted by date
data.sort_values('Date', inplace=True)

# Extract returns
returns = data['Daily Return'].dropna() * 100  # Percentage returns

data = vol_loader.load_ticker_data(TICKER)

data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
target = data['Volatility'].values  # Volatility as the continuous target
dates = data['Date'].values
# 2) train-test split including dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    features, target, dates, train_size=SPLIT_SIZE, shuffle=False
)
"""
# Extract returns
returns = data['Daily Return'].dropna() * 100  # Percentage returns

# Split into train and test sets
train_size = int(len(returns) * SPLIT_SIZE)
train_returns = returns[:train_size]
test_returns = returns[train_size:]

# Fit GARCH(1,1) model on training data
garch_model = arch_model(train_returns, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
garch_fit = garch_model.fit(disp='off')

# Forecast volatility for test set
garch_forecast = garch_fit.forecast(horizon=1, reindex=False)
predicted_volatility_garch = garch_forecast.variance.values[-len(test_returns):].flatten()
predicted_volatility_garch = np.sqrt(predicted_volatility_garch)  # Convert variance to standard deviation

# Actual volatility
actual_volatility = data['Volatility'][train_size + SEQUENCE_LENGTH:].values  # Adjust based on your sequence length

# Evaluate GARCH performance
mse_garch = mean_squared_error(actual_volatility, predicted_volatility_garch)
mae_garch = mean_absolute_error(actual_volatility, predicted_volatility_garch)

print(f"GARCH Model - MSE: {mse_garch}, MAE: {mae_garch}")

def plot():
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test, actual_volatility, label='Actual Volatility', color='black')
    plt.plot(dates_test, predicted_volatility_garch, label='GARCH Predicted Volatility', color='red')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('GARCH Predicted vs Actual Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import vol_loader
from sklearn.model_selection import train_test_split


SPLIT_SIZE = 0.7
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20

# Load your data
data = vol_loader.load_ticker_data(TICKER)

# Ensure data is sorted by date
data.sort_values('Date', inplace=True)

# Extract returns
returns = data['Daily Return'].dropna() * 100  # Percentage returns

data = vol_loader.load_ticker_data(TICKER)

data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

features = data[['Daily Return']].values  # Use 'Daily Return' as input feature
target = data['Volatility'].values  # Volatility as the continuous target
dates = data['Date'].values
# 2) train-test split including dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    features, target, dates, train_size=SPLIT_SIZE, shuffle=False
)

# Fit GARCH(1,1) model on training data
garch_model = arch_model(X_train, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
garch_fit = garch_model.fit(disp='off')

# Forecast volatility for test set
garch_forecast = garch_fit.forecast(horizon=1, reindex=False)
predicted_volatility_garch = garch_forecast.variance.values[-len(X_test):].flatten()
predicted_volatility_garch = np.sqrt(predicted_volatility_garch)  # Convert variance to standard deviation

# Actual volatility
actual_volatility = data['Volatility'][train_size + SEQUENCE_LENGTH:].values  # Adjust based on your sequence length

# Evaluate GARCH performance
mse_garch = mean_squared_error(actual_volatility, predicted_volatility_garch)
mae_garch = mean_absolute_error(actual_volatility, predicted_volatility_garch)

print(f"GARCH Model - MSE: {mse_garch}, MAE: {mae_garch}")

def plot():
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test, actual_volatility, label='Actual Volatility', color='black')
    plt.plot(dates_test, predicted_volatility_garch, label='GARCH Predicted Volatility', color='red')
    plt.plot(dates_test, y_pred_inverse, label='LSTM Predicted Volatility', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('GARCH vs LSTM - Predicted vs Actual Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
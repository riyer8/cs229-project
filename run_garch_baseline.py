"""
run_garch_baseline_corrected.py
-------------------
Run GARCH model for time series volatility forecasting with proper evaluation.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data import vol_loader  # Ensure this module is correctly implemented

# Constants
TICKER = 'AAPL'
SPLIT_SIZE = 0.7
WINDOW_SIZE = 20  # Window size for realized volatility calculation

###################### Functionality

# 1) Load data
data = vol_loader.load_ticker_data(TICKER)
data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

# Check if 'Date' column exists; if not, use the index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
else:
    data['Date'] = data.index

# Ensure 'Daily Return' is present
if 'Daily Return' not in data.columns:
    raise ValueError("The 'Daily Return' column is missing from the data.")

# Prepare target variable (Raw Returns)
returns = data['Daily Return'].values  # Assuming 'Daily Return' is available

# 1a) Calculate Realized Volatility using a rolling window
realized_volatility = pd.Series(returns).rolling(window=WINDOW_SIZE).std().dropna().values

# Shift realized_volatility by one to align with aligned_returns
realized_volatility = realized_volatility[1:]

# Align returns to match realized_volatility
aligned_returns = returns[WINDOW_SIZE:]
dates_aligned = data['Date'].values[WINDOW_SIZE:]

# 2) Train-test split considering the rolling window
train_size = int(len(aligned_returns) * SPLIT_SIZE)
y_train = aligned_returns[:train_size]
y_test = aligned_returns[train_size:]
realized_vol_train = realized_volatility[:train_size]
realized_vol_test = realized_volatility[train_size:]
dates_train = dates_aligned[:train_size]
dates_test = dates_aligned[train_size:]

# Verification: Print the lengths
print(f"Train size: {len(y_train)}")
print(f"Test size: {len(y_test)}")
print(f"Realized Vol Train size: {len(realized_vol_train)}")
print(f"Realized Vol Test size: {len(realized_vol_test)}")

# 3) Fit GARCH model on training data
model = arch_model(y_train, vol='Garch', p=1, q=1, rescale=False)
garch_fit = model.fit(disp="off")

# Print the summary of the GARCH model for diagnostics
print(garch_fit.summary())

# Check if the model fit was successful
if garch_fit.convergence_flag != 0:
    raise ValueError("Model fitting failed to converge.")

# 4) Rolling Forecasts for Volatility
predicted_vol = []

# Initialize the data for rolling forecasts
rolling_data = y_train.copy().tolist()

# Iterate over the test set
for i in range(len(y_test)):
    # Fit the GARCH model on the current rolling data
    model = arch_model(rolling_data, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = model.fit(disp="off")
    
    # Check convergence
    if garch_fit.convergence_flag != 0:
        raise ValueError(f"Model fitting failed to converge at step {i}.")
    
    # Forecast one step ahead
    forecast = garch_fit.forecast(horizon=1, reindex=False)
    
    # Extract the variance forecast
    variance_forecast = forecast.variance.iloc[-1, 0]
    
    # Convert variance to volatility
    vol_forecast = np.sqrt(variance_forecast)
    predicted_vol.append(vol_forecast)
    
    # Append the actual observed return to the rolling data for the next iteration
    rolling_data.append(y_test[i])
    
    # Optional: Print progress every 100 steps
    if (i+1) % 100 == 0:
        print(f"Completed {i+1} out of {len(y_test)} forecasts.")

# Convert predicted_vol to a NumPy array for consistency
predicted_vol = np.array(predicted_vol)

# 5) Evaluate the model
# Check lengths
print(f"Length of realized_vol_test: {len(realized_vol_test)}")
print(f"Length of predicted_vol: {len(predicted_vol)}")

# Assert to ensure alignment
assert len(realized_vol_test) == len(predicted_vol), \
    f"Length mismatch: realized_vol_test has {len(realized_vol_test)} samples, predicted_vol has {len(predicted_vol)} samples."

mse = mean_squared_error(realized_vol_test, predicted_vol)
mae = mean_absolute_error(realized_vol_test, predicted_vol)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# 6) Plot Predictions vs Actual Realized Volatility
def plot_preds(dates, actual, predicted):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label="Realized Volatility", color='blue')
    plt.plot(dates, predicted, label="Predicted Volatility", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("GARCH Model - Predicted vs. Realized Volatility")
    plt.legend()

    # Improve date formatting on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    plt.tight_layout()
    plt.show()

# Align the dates correctly for plotting
plot_preds(dates_test, realized_vol_test, predicted_vol)

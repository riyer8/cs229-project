"""
run_garch_model.py
-------------------
Run GARCH model for time series volatility forecasting
"""
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data import vol_loader

# consts
TICKER = 'AAPL'
SPLIT_SIZE = 0.7

###################### functionality

# 1) Load data
data = vol_loader.load_ticker_data(TICKER)
data.dropna(inplace=True)  # Drop NaNs resulting from rolling calculations

# Check if 'Date' column exists; if not, use the index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
else:
    data['Date'] = data.index

# Prepare target variable (Volatility)
returns = data['Daily Return'].values  # Assuming 'Daily Return' is available

# Rescale to improve convergence
y_rescaled = 1000 * returns  

# 2) Train-test split
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    returns, y_rescaled, data['Date'].values, train_size=SPLIT_SIZE, shuffle=False
)

# 3) Fit GARCH model on training data
model = arch_model(y_train, vol='Garch', p=1, q=1)
garch_fit = model.fit(disp="off")

# Print the summary of the GARCH model for diagnostics
print(garch_fit.summary())

# Check if the model fit was successful
if garch_fit.convergence_flag == 0:
    # 4) Generate predictions for the test set
    pred_volatility = garch_fit.forecast(start=len(y_train), reindex=False)

    # Check if the forecast provides valid variance values
    if pred_volatility.variance.shape[0] > 0 and np.any(pred_volatility.variance.values):
        predicted_vol = np.sqrt(pred_volatility.variance.values[-1, :])  # Get predicted volatilities
    else:
        raise ValueError("Forecast failed: no valid variance values to predict.")
else:
    raise ValueError("Model fitting failed to converge.")

# 5) Evaluate the model
mse = mean_squared_error(y_test, predicted_vol[:len(y_test)])
mae = mean_absolute_error(y_test, predicted_vol[:len(y_test)])

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# 6) Plot Predictions vs Actual
def plot_preds(dates, actual, predicted):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label="Actual Volatility", color='blue')
    plt.plot(dates, predicted, label="Predicted Volatility", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("GARCH Model - Predicted vs. Actual Volatility")
    plt.legend()

    # Improve date formatting on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    plt.tight_layout()
    plt.show()

# Align the dates correctly for plotting
plot_preds(dates_test[:len(predicted_vol)], y_test, predicted_vol[:len(y_test)])
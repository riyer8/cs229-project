import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data.ticker_settings import ALL_TICKERS
from data.vol_loader import generate_ticker_vol, plot_historical_volatility

# Parameters
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = ALL_TICKERS[0]  # Adjust as needed
ENABLE_PLOTTING = False  # Set to True if you want to plot volatility

data = generate_ticker_vol(TICKER, START_DATE, END_DATE)

data.dropna(inplace=True)

# Prepare Features and Target
features = data[['Daily Return', 'Volatility']]
target = data['Volatility']

# Split Data into Training and Testing Sets
train_size = int(0.7 * len(features))
X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² (Coefficient of Determination):", r2)

# Plot Volatility if enabled
if ENABLE_PLOTTING:
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    plot_historical_volatility(TICKER, START_DATE, END_DATE, train_data, test_data)
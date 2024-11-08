import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data.ticker_settings import ALL_TICKERS
from data.vol_loader import generate_ticker_vol, plot_historical_volatility

# Parameters
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = ALL_TICKERS[0]  # Adjust as needed
ENABLE_PLOTTING = False  # Set to True if you want to plot volatility

# Load Data and Calculate Volatility using vol_loader.py
data = generate_ticker_vol(TICKER, START_DATE, END_DATE)

# Classify High vs. Low Volatility
# Define high vs. low volatility based on the 75th percentile threshold
vol_threshold = data['Volatility'].quantile(0.75)
data['Volatility Class'] = np.where(data['Volatility'] > vol_threshold, 1, 0)  # 1 for high, 0 for low

# Drop rows with NaN values (if any) resulting from the rolling calculation
data.dropna(inplace=True)

# Prepare Features and Target
features = data[['Daily Return', 'Volatility']]
target = data['Volatility Class']

# Split Data into Training and Testing Sets
train_size = int(0.7 * len(features))
X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Plot Volatility if enabled
if ENABLE_PLOTTING:
    plot_historical_volatility(TICKER, START_DATE, END_DATE)
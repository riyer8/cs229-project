import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from portfolioInfo import ALL_TICKERS

WINDOW_DAYS = 100 # smoothness factor

START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = ALL_TICKERS[0]
ENABLE_PLOTTING = False

def historical_volatility(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100
    
def plot_historical_volatility(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Volatility'], label='Volatility', color='blue')
    plt.title(f'Historical Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.legend()
    plt.show()

ticker = TICKER

data_full = yf.download(ticker, period="max")
ipo_date = data_full.index.min().strftime('%Y-%m-%d')
start_date = max(ipo_date, START_DATE)
end_date = END_DATE

historical_volatility(ticker, start_date, end_date)

if ENABLE_PLOTTING:
    plot_historical_volatility(ticker, start_date, end_date)
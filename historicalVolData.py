"""
This script calculates and optionally plots the historical volatility for a single stock ticker 
from a predefined list of tickers (`ALL_TICKERS`). The code specifically selects the first ticker 
in the list (`ALL_TICKERS[0]`) and performs the following operations:

1. Downloads historical stock price data from Yahoo Finance for the selected ticker.
2. Calculates the daily returns and the 100-day rolling annualized volatility of the stock.
3. If `ENABLE_PLOTTING` is set to True, it plots the calculated volatility over time.

Note:
- The script is currently set to process and plot data for only the first ticker in `ALL_TICKERS`.
- To enable plotting, change `ENABLE_PLOTTING` to True.
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from portfolioInfo import ALL_TICKERS

WINDOW_DAYS = 100 # smoothness factor

START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
TICKER = ALL_TICKERS[0]
ENABLE_PLOTTING = True

def historical_volatility(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100
    return data

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
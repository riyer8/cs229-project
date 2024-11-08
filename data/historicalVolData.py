"""
historicalVolData.py
-------------------
Utility functions to save and load ticker volatility data, as well as plot the historical volatility for a single stock ticker

Volatility calculation operations:

1. Downloads historical stock price data from Yahoo Finance for the selected ticker.
2. Calculates the daily returns and the 100-day rolling annualized volatility of the stock.
"""

import os
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from portfolioInfo import ALL_TICKERS


# constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"
WINDOW_DAYS = 100  # smoothness factor
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
DATA_DIR = "datasets"


def load_ticker_data(ticker) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{ticker}_volatility.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for {ticker}. Please generate it first.")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Data for {ticker} loaded from {file_path}")
    return data


def generate_ticker_vol(ticker, start_date, end_date) -> None:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100
    print(data.head())
    file_path = os.path.join(DATA_DIR, f"{ticker}_volatility.csv")
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")


def plot_historical_volatility(ticker):
    data = load_ticker_data(ticker)
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Volatility'], label='Volatility', color='blue')
    plt.title(f'Historical Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.legend()
    plt.show()


def download_data():
    for ticker in ALL_TICKERS[:1]:
        generate_ticker_vol(ticker, START_DATE, END_DATE)


if __name__ == '__main__':
    download_data()
    # data = load_ticker_data("AAPL")
    # print(data.head())

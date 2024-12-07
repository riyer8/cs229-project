"""
vol_loader.py
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
from data.ticker_settings import ALL_TICKERS

# Constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"
WINDOW_DAYS = 21  # window size, following typical
START_DATE = '2006-01-01'  # time frame: 2006 to 2020 to match sentiment data
END_DATE = '2020-12-31'
DATA_DIR_LOAD = "data/datasets"  # full path when reading from outside directory
DATA_DIR_SAVE = "datasets"


def load_ticker_data(ticker) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR_LOAD, f"{ticker}_volatility.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for {ticker}. Please generate it first.")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Data for {ticker} loaded from {file_path}")
    return data


def generate_ticker_dataset(ticker, start_date, end_date) -> None:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.droplevel('Ticker')
    data.columns.name = None
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.dropna(inplace=True)

    # derived indicators
    data['Daily Variation'] = (data['High'] - data['Low']) / data['Open']
    data['7-Day SMA'] = data['Adj Close'].rolling(window=7).mean()
    data['7-Day STD'] = data['Adj Close'].rolling(window=7).std()
    data['SMA + 2STD'] = data['7-Day SMA'] + 2 * data['7-Day STD']
    data['SMA - 2STD'] = data['7-Day SMA'] - 2 * data['7-Day STD']
    data['High-Close'] = (data['High'] - data['Close']) / data['Open']
    data['Low-Open'] = (data['Low'] - data['Open']) / data['Open']
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    data['14-Day EMA'] = data['Adj Close'].ewm(span=14, adjust=False).mean()
    data['Close % Change'] = data['Adj Close'].pct_change()
    data['Close Change'] = data['Adj Close'].diff()

    # RSI
    delta = data['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26

    # Stochastic Oscillator
    data['Stochastic Oscillator'] = (data['Adj Close'] - data['Low'].rolling(window=14).min()) / \
                                    (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())

    # ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Adj Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Adj Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # calculating ADX and DMI
    data['+DM'] = data['High'].diff()
    data['-DM'] = data['Low'].diff()
    data['+DM'] = np.where((data['+DM'] > 0) & (data['+DM'] > data['-DM']), data['+DM'], 0)
    data['-DM'] = np.where((data['-DM'] > 0) & (data['-DM'] > data['+DM']), data['-DM'], 0)

    data['TR'] = np.maximum(data['High'] - data['Low'],
                            np.maximum(abs(data['High'] - data['Adj Close'].shift(1)),
                                       abs(data['Low'] - data['Adj Close'].shift(1))))

    # smooth +DM, -DM, and TR using Wilder's smoothing
    data['Smoothed +DM'] = data['+DM'].rolling(window=WINDOW_DAYS).mean()
    data['Smoothed -DM'] = data['-DM'].rolling(window=WINDOW_DAYS).mean()
    data['Smoothed TR'] = data['TR'].rolling(window=WINDOW_DAYS).mean()

    # DMI
    data['+DI'] = (data['Smoothed +DM'] / data['Smoothed TR']) * 100
    data['-DI'] = (data['Smoothed -DM'] / data['Smoothed TR']) * 100

    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100

    # ADX
    data['ADX'] = data['DX'].rolling(window=WINDOW_DAYS).mean()

    # save data to csv
    columns_to_drop = ['+DM', '-DM', 'TR', 'Smoothed +DM', 'Smoothed -DM', 'Smoothed TR', 'DX']
    data.drop(columns=columns_to_drop, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(data.head())
    file_path = os.path.join(DATA_DIR_SAVE, f"{ticker}_volatility.csv")
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")
    return data


def plot_historical_volatility(ticker, start_date=None, end_date=None):
    data = load_ticker_data(ticker)
    if start_date or end_date:
        data = data.loc[start_date:end_date]

    data.dropna(subset=['Volatility'], inplace=True)

    # Plot volatility
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Volatility'], label='Volatility', color='blue')
    plt.title(f'Historical Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.legend()
    plt.show()


def download_data():
    for ticker in ALL_TICKERS:
        print(f"Starting ticker {ticker}")
        generate_ticker_dataset(ticker, START_DATE, END_DATE)


if __name__ == '__main__':
    download_data()


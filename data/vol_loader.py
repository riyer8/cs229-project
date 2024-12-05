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
from ticker_settings import ALL_TICKERS
from historical_info import get_dividend_price_ratio, get_earnings_price_ratio, \
    get_book_to_market_ratio, get_stock_variance, get_treasury_bill_rate

# Constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"
WINDOW_DAYS = 100  # smoothness factor
START_DATE = '2003-01-01'
END_DATE = '2023-12-31'
DATA_DIR_LOAD = "data/datasets"  # full path when reading from outside directory
DATA_DIR_SAVE = "datasets"


def load_ticker_data(ticker) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR_LOAD, f"{ticker}_volatility.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for {ticker}. Please generate it first.")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Data for {ticker} loaded from {file_path}")
    return data


def generate_ticker_vol(ticker, start_date, end_date) -> None:
    # Download historical stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=WINDOW_DAYS).std() * np.sqrt(252) * 100
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.dropna(inplace=True)

    # Add additional metrics
    dividend_price_ratio = get_dividend_price_ratio(data)
    earnings_price_ratio = get_earnings_price_ratio(data)
    book_to_market_ratio = get_book_to_market_ratio(ticker)
    stock_variance = get_stock_variance(data)
    risk_free_rate = get_treasury_bill_rate()

    # Add metrics as columns
    data['Dividend Price Ratio'] = dividend_price_ratio
    data['Earnings Price Ratio'] = earnings_price_ratio
    data['Book to Market Ratio'] = book_to_market_ratio
    data['Stock Variance'] = stock_variance
    data['Risk-Free Rate'] = risk_free_rate

    print(data.head())

    # Save data to CSV
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
        generate_ticker_vol(ticker, START_DATE, END_DATE)


if __name__ == '__main__':
    download_data()
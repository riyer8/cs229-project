import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader as pdr

def get_dividend_price_ratio(data):
    """
    Calculate the dividend-to-price ratio using the provided data.
    """
    if 'Dividends' in data.columns:
        dividend_yield = data['Dividends'].sum() / data['Close'].iloc[-1]
        dividend_price_ratio = dividend_yield / data['Close'].iloc[-1]
    else:
        print(f"Dividends data not available.")
        dividend_price_ratio = np.nan
    return dividend_price_ratio


def get_earnings_price_ratio(data):
    """
    Calculate the earnings-to-price ratio using the provided data.
    """
    if 'Earnings' in data.columns:
        earnings_price_ratio = data['Earnings'].iloc[-1] / data['Close'].iloc[-1]
    else:
        print(f"Earnings data not available.")
        earnings_price_ratio = np.nan
    return earnings_price_ratio


def get_book_to_market_ratio(ticker):
    """
    Fetch the book-to-market ratio for a given ticker using price-to-book ratio.
    """
    ticker_data = yf.Ticker(ticker)
    try:
        price_to_book = ticker_data.info['priceToBook']  # Access price-to-book ratio
        book_to_market = 1 / price_to_book if price_to_book else np.nan
    except KeyError:
        print(f"Price-to-Book ratio not available for {ticker}.")
        book_to_market = np.nan  # Handle missing data
    return book_to_market


def get_stock_variance(data):
    """
    Calculate the stock variance using the provided data.
    """
    returns = np.log(data['Close'] / data['Close'].shift(1))
    stock_variance = np.var(returns)
    return stock_variance


def get_treasury_bill_rate():
    """
    Fetch the current 1-month Treasury bill rate from FRED.
    """
    t_bill_data = pdr.DataReader('DGS1MO', 'fred', start='2023-01-01')
    return t_bill_data.iloc[-1].values[0]  # Latest rate
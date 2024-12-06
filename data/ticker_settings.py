"""
ticker_settings.py
--------
Defines stock ticker constants
"""

# Benchmarks and Indices
BENCHMARKS_AND_INDICES = [
    '^DJI',  # DOW JONES
    'SPY',  # S&P 500 ETF
    'QQQ',  # NASDAQ 100 ETF
]

# Technology Companies
TECHNOLOGY_COMPANIES = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META'
]

# Semiconductor Companies
SEMICONDUCTOR_COMPANIES = [
    'NVDA', 'ASML', 'MU', 'AMAT', 'LRCX', 'AVGO', 'QCOM', 'TXN'
]

# Consumer Discretionary Companies
CONSUMER_DISCRETIONARY_COMPANIES = [
    'TSLA', 'HD', 'NKE', 'MCD', 'DIS'
]

# Energy Companies
ENERGY_COMPANIES = [
    'XOM', 'CVX', 'COP', 'SLB'
]

# Healthcare Companies
HEALTHCARE_COMPANIES = [
    'JNJ', 'PFE', 'AMGN', 'ABBV'
]

# Financial Companies
FINANCIAL_COMPANIES = [
    'JPM', 'GS', 'V', 'MA'
]

# Consumer Staples Companies
CONSUMER_STAPLES_COMPANIES = [
    'PG', 'KO', 'PEP', 'WMT'
]

# Communication Companies
COMMUNICATION_COMPANIES = [
    'T', 'VZ', 'CMCSA'
]

# Utilities Companies
UTILITIES_COMPANIES = [
    'DUK', 'SO', 'EXC'
]

# Materials Companies
MATERIALS_COMPANIES = [
    'DOW', 'MLM', 'VMC', 'NEM'
]

# Travel Companies
TRAVEL_COMPANIES = [
    'DAL', 'UAL', 'AAL', 'RCL'
]

# Combine all tickers into a single list and remove duplicates
ALL_TICKERS = sorted(list(set(
    BENCHMARKS_AND_INDICES + TECHNOLOGY_COMPANIES + SEMICONDUCTOR_COMPANIES +
    CONSUMER_DISCRETIONARY_COMPANIES + ENERGY_COMPANIES + HEALTHCARE_COMPANIES +
    FINANCIAL_COMPANIES + CONSUMER_STAPLES_COMPANIES + COMMUNICATION_COMPANIES +
    UTILITIES_COMPANIES + MATERIALS_COMPANIES + TRAVEL_COMPANIES
)))


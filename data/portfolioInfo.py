"""
portfolioInfo.py
--------
Defines stock ticker constants
"""

# Technology Companies
TECHNOLOGY_COMPANIES = [
    'AMD', 'IBM', 'CRM', 'HPE', 'DELL', 'ADBE', 'AI', 'SHOP', 'TWLO', 'SPOT', 'Z',
    'DDOG', 'PINS', 'AMZN', 'META', 'CSCO', 'ORCL', 'ANET', 'PANW', 'DLTR', 'HPQ',
    'CRWD', 'MDB', 'GTLB', 'SQ', 'ZM', 'PYPL', 'RBLX', 'CFLT', 'ADSK', 'KEYS', 'NOW',
    'SOFI', 'RIVN', 'QS', 'TEAM', 'ROKU', 'AFRM', 'MNDY', 'GOOGL', 'MSFT', 'AAPL',
    'INTC', 'BABA', 'PLTR', 'COST', 'SNOW', 'FSLY', 'TGTX', 'HUBS'
]

# Semiconductor Companies
SEMICONDUCTOR_COMPANIES = [
    'NVDA', 'ASML', 'MU', 'AMAT', 'LRCX', 'AVGO', 'QCOM', 'ON', 'SMCI', 'CRUS',
    'TXN', 'SWKS'
]

# Consumer Discretionary Companies
CONSUMER_DISCRETIONARY_COMPANIES = [
    'TSLA', 'HD', 'NKE', 'MCD', 'DIS', 'LULU', 'ETSY', 'ULTA', 'CMG', 'TGT',
    'BBY', 'MAR', 'RCL', 'LUV', 'PTON', 'GRWG'
]

# Energy Companies
ENERGY_COMPANIES = [
    'XOM', 'CVX', 'COP', 'SLB', 'ENPH'
]

# Healthcare Companies
HEALTHCARE_COMPANIES = [
    'JNJ', 'PFE', 'MRNA', 'GILD', 'AMGN', 'ABBV'
]

# Financial Companies
FINANCIAL_COMPANIES = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'V', 'MA', 'AXP', 'MS', 'BK'
]

# Consumer Staples Companies
CONSUMER_STAPLES_COMPANIES = [
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CPB', 'MDLZ', 'SJM'
]

# Communication Companies
COMMUNICATION_COMPANIES = [
    'T', 'VZ', 'TMUS', 'CMCSA', 'NFLX', 'TTD'
]

# Utilities Companies
UTILITIES_COMPANIES = [
    'DUK', 'SO', 'EXC', 'AEP', 'NEE'
]

# Materials Companies
MATERIALS_COMPANIES = [
    'DOW', 'MLM', 'VMC', 'NEM', 'FCX'
]

# Travel Companies
TRAVEL_COMPANIES = [
    'UAL', 'AAL', 'DAL', 'CCL', 'RCL', 'EXPE', 'BKNG'
]

ALL_TICKERS = sorted(list(set((
    SEMICONDUCTOR_COMPANIES + TECHNOLOGY_COMPANIES + CONSUMER_DISCRETIONARY_COMPANIES 
    + ENERGY_COMPANIES + HEALTHCARE_COMPANIES + FINANCIAL_COMPANIES 
    + CONSUMER_STAPLES_COMPANIES + COMMUNICATION_COMPANIES + UTILITIES_COMPANIES 
    + MATERIALS_COMPANIES + TRAVEL_COMPANIES
))))
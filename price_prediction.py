from statsmodels.regression.rolling import RollingOLS # estimates relationship b/w dep & indep var using min square diff over a rolling window
import pandas_datareader.data as web # takes web info and stores as dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # calculations
import datetime as dt
import yfinance as yf # stock data
import statsmodels.api as sm # stat models
import pandas_ta # technical indicators calculator
import warnings

warnings.filterwarnings('ignore') # annoying :(

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2024-02-01'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*10)

df = yf.download(tickers=symbols_list, start=start_date, end=end_date)

print(df)
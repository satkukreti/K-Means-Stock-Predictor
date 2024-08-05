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

# collect sp500 stock data over the past decade, leaving a time gap for testing
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2024-02-01'
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=10)

df = yf.download(tickers=symbols_list, start=start_date, end=end_date)
df = df.stack() # multi indexed

df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# calculate technical indicators
# Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, Dollar Volume

df['garman-klass_vol'] = (0.5 * (np.log(df['high'] / df['low'])**2)) - (2 * np.log(2) - 1) * (np.log(df['adj close'] / df['open'])**2)
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x : pandas_ta.rsi(close=x, length=20))
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std()) # normalize it

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std()) # normalize it

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
df['dollar_vol'] = (df['adj close']*df['volume'])/1e6 # divide by a million

# normalizing will help in each factor contributing equally to the centroid distance calculation
# aggregate to a monthly level and filter top 150 liquid stocks

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_vol', 'volume', 'open', 'high', 'low', 'close']]
data = (pd.concat([df.unstack('ticker')['dollar_vol'].resample('M').mean().stack('ticker').to_frame('dollar_vol'), 
                   df.unstack()[last_cols].resample('M').last().stack('ticker')], axis=1)).dropna()

# calculate 5 year rolling avg dollar vol
data['dollar_vol'] = data['dollar_vol'].unstack('ticker').rolling(window=5*12, min_periods=1).mean().stack()
data['dollar_vol_rank'] = data.groupby('date')['dollar_vol'].rank(ascending=False)

data = data[data['dollar_vol_rank'] < 150].drop(['dollar_vol', 'dollar_vol_rank'], axis=1)

# calculate monthly returns for different time horizons as features
# lags for 1, 2, 3, 6, 9, and 12 months

def calculate_returns(df):
    outlier_cutoff = 0.005 # 99.5 percentile
    lags = [1, 2, 3, 6, 9, 12]
    
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                               .pct_change(lag)
                               .pipe(lambda x : x.clip(lower=x.quantile(outlier_cutoff),
                                                        upper=x.quantile(1-outlier_cutoff)))
                               .add(1)
                               .pow(1/lag)
                               .sub(1))
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# fama-french factors for better risk evaluation

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                             start='2015')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp() # converts from period to datetime index
factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date' 

# we can now compare the factor to the 1m returns to calculate the beta

factor_data = factor_data.join(data['return_1m']).sort_index()

observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker')
                          .isin(valid_stocks.index)] # only keeping stocks with >= 10 months of data for rolling factor betas calcs

betas = factor_data.groupby(level=1,
                            group_keys=False).apply(lambda x : RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
               .fit(params_only=True)
               .params.drop('const', axis=1))

# shifting the betas as we would only know them at the end of the month
# combine them with the other features

data = data.join(betas.groupby('ticker').shift())

# fill in Nan values with mean

factors = ['Mkt-RF', 'SMB',	'HML', 'RMW', 'CMA']
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x : x.fillna(x.mean()))

data = data.dropna()
data = data.drop('adj close', axis = 1)

# apply machine learning model to decide on what stocks our porfolio consists of
# training it based on having a long portfolio
# K-means clustering algo to group stocks based on features
#  K = 4
# strategy will be based on rsi value, so which stocks have the highest momentum in the previous month

from sklearn.cluster import KMeans

target_rsi_vals = [30, 45, 55, 70]
initial_centroids = np.zeros((len(target_rsi_vals), 18)) # centers and num features
initial_centroids[:, 6] = target_rsi_vals

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                          random_state=0, # random seed
                          init=initial_centroids).fit(df).labels_ # clusters created based on pre-defined centers
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

# create a scatter plot visualization

def plot_clusters(df):
    cluster0 = df[df['cluster']==0]
    cluster1 = df[df['cluster']==1]
    cluster2 = df[df['cluster']==2]
    cluster3 = df[df['cluster']==3]

    plt.scatter(cluster0.iloc[:,0], cluster0.iloc[:,6], color='red', label='cluster 0')
    plt.scatter(cluster1.iloc[:,0], cluster1.iloc[:,6], color='green', label='cluster 1')
    plt.scatter(cluster2.iloc[:,0], cluster2.iloc[:,6], color='blue', label='cluster 2')
    plt.scatter(cluster3.iloc[:,0], cluster3.iloc[:,6], color='yellow', label='cluster 3')

    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist(): # display on a monthly basis
    g = data.xs(i, level=0)
    plt.title(f'Date {i}')
    plot_clusters(g)

# select stocks based on cluster 3, high rsi should outperform even in later months
filtered_df = data[data['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level=1)

# use these values for the stocks we want to invest in the next month
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

# create a dict to store the name of the stocks we want
dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()


# form a portfolio based on EfficientFrontier max sharpe ratio
# maximize weights
# note: sharpe ratio helps us calculate the highest amount of return for the lowest expected risk

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0): # supply full year of prices
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252) # expected returns
    cov = risk_models.sample_cov(prices=prices, frequency=252) # covariance

    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound,.1), # diversify by capping the weights
                           solver='SCS')
    weights = ef.max_sharpe()

    return ef.clean_weights()

# get the past year of stock info
stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                    end=data.index.get_level_values('date').unique()[-1])

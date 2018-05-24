import qtrader

import os

import pandas as pd

# fetch data
sp500, prices, returns = qtrader.envs.data_loader.Finance.SP500(
    return_prices_returns=True)
# make if not there 'db' folder
if not os.path.exists('db'):
    os.makedirs('db')
# store data
sp500.to_csv('db/sp500.csv')
prices.to_csv('db/prices.csv')
returns.to_csv('db/returns.csv')

# remove from score
del sp500
del prices
del returns

# read data
sp500 = pd.read_csv('db/sp500.csv', index_col=0, header=0)
prices = qtrader.envs.data_loader.Finance.Prices(
    sp500.index.tolist(), csv='db/prices.csv')
returns = qtrader.envs.data_loader.Finance.Returns(
    sp500.index.tolist(), csv='db/returns.csv')

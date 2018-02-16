import pandas as pd
import pandas_datareader.data as web

TICKERS = ['AAPL', 'VOD', 'MSFT', 'GE']

df = pd.DataFrame(columns=TICKERS)

for ticker in TICKERS:
    df[ticker] = web.DataReader(
        'WIKI/%s' % ticker, data_source='quandl', start='2015')['AdjClose']

df.sort_index(ascending=True).to_csv('tests/tmp/data/prices.csv')

from qtrader.envs.data_loader import Finance

TICKERS = ['AAPL', 'VOD', 'MSFT', 'GE']
CSV_FILE = 'tests/tmp/data/prices.csv'
start_date = '2010-01-01'

df = Finance.Prices(TICKERS, start_date=start_date)

df.sort_index(ascending=True).to_csv(CSV_FILE)

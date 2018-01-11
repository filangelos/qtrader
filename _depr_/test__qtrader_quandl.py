# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing
import numpy as np

import qtrader as qt

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

start_date = '2015-01-01'
end_date = '2017-01-01'

universe = ['AAPL', 'GOOGL', 'MSFT']

qt.data.Market.start_date = start_date
qt.data.Market.end_date = end_date
qt.data.Market.source = 'yahoo'

returns = qt.data.Market.Returns(universe)

returns[universe[0]].hist(bins=50)
plt.savefig('tests/tmp/test__qtrader_quandl_returns.pdf',
            format='pdf', dpi=300)

plt.figure()

prices = qt.data.Market.Prices(universe)

prices[universe[0]].plot()
plt.savefig('tests/tmp/test__qtrader_quandl_prices.pdf', format='pdf', dpi=300)

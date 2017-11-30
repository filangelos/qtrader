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

universe = ['AAPL', 'GOOGL', 'MSFT', ]

qt.data.Quandl.start_date = start_date
qt.data.Quandl.end_date = end_date

data = qt.data.Quandl.Returns(universe)

data[universe[0]].hist(bins=50)
plt.savefig('tests/tmp/test__qtrader_quandl.pdf', format='pdf', dpi=300)

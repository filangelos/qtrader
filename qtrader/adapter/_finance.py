from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import pandas as pd

# market data bundler
import pandas_datareader.data as web


class Finance:
    """Market Data Wrapper."""
    _close_col = {'quandl': 'Close', 'yahoo': 'Adj Close'}

    @classmethod
    def _get(cls, ticker, **kwargs):
        """Helper method for `web.DataReader`.
        Parameters
        ----------
        ticker: str
            Ticker name
        **kwargs: dict
            Arguments for `quandl.get`
        Returns
        -------
        df: pandas.DataFrame
            Table of prices for `ticker`
        """
        return web.DataReader(ticker, **kwargs)

    @classmethod
    def Returns(cls, tickers, start_date=None, end_date=None, source='quandl'):
        """Get daily returns for `tickers`.
        Parameters
        ----------
        tickers: list
            List of ticker names
        Returns
        -------
        df: pandas.DataFrame
            Table of Returns of Adjusted Close prices for `tickers`
        """
        return cls.Prices(tickers, start_date, end_date, source).pct_change()[1:]

    @classmethod
    def Prices(cls, tickers, start_date=None, end_date=None, source='quandl'):
        """Get daily prices for `tickers`.
        Parameters
        ----------
        tickers: list
            List of ticker names
        Returns
        -------
        df: pandas.DataFrame | pandas.Series
            Table of Adjusted Close prices for `tickers`
        """
        df = pd.DataFrame.from_dict({ticker: cls._get(ticker, data_source=source, start=start_date,
                                                      end=end_date)[cls._close_col[source]] for ticker in tickers})
        if len(df.columns) == 1:
            df = df[df.columns[0]]
        return df

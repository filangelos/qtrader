# library logger
from qtrader.framework.logger import logger

# scientific computing
import numpy as np
import pandas as pd

import os
import typing

# market data provider
import quandl
quandl.ApiConfig.api_key = os.environ.get('QUANDL_API_KEY')


class Finance:
    """Market Data Wrapper."""
    _col = 'Adj. Close'

    @classmethod
    def _get(cls,
             ticker: str,
             **kwargs) -> typing.Optional[pd.DataFrame]:
        """Helper method for `quandl.get`.

        Parameters
        ----------
        ticker: str
            Ticker name
        **kwargs: dict
            Arguments for `quandl.get`
        Returns
        -------
        df: pandas.DataFrame
            Market data for `ticker`
        """
        try:
            return quandl.get('WIKI/%s' % ticker, **kwargs)
        except:
            logger.warn('failed to fetch market data for %s' % ticker)
            return None

    @classmethod
    def _csv(cls,
             root: str,
             tickers: typing.Union[str, typing.List[str]]):
        """Helper method for loading prices from csv files.

        Parameters
        ----------
        root: str
            Path of csv file
        ticker: str
            Ticker name
        """
        df = pd.read_csv(root, index_col='Date',
                         parse_dates=True).sort_index(ascending=True)
        union = [ticker for ticker in tickers if ticker in df.columns]
        return df[union]

    @classmethod
    def Returns(cls,
                tickers: typing.List[str],
                start_date: str = None,
                end_date: str = None,
                freq: str = 'B',
                csv: typing.Optional[str] = None):
        """Get daily returns for `tickers`.

        Parameters
        ----------
        tickers: list
            List of ticker names
        freq: str
            Resampling frequency
        Returns
        -------
        df: pandas.DataFrame
            Table of Returns of Adjusted Close prices for `tickers`
        """
        if isinstance(csv, str):
            return cls._csv(csv, tickers).loc[start_date:end_date]
        else:
            return cls.Prices(tickers,
                              start_date,
                              end_date,
                              freq).pct_change()[1:]

    @classmethod
    def Prices(cls,
               tickers: typing.List[str],
               start_date: str = None,
               end_date: str = None,
               freq: str = 'B',
               csv: typing.Optional[str] = None):
        """Get daily prices for `tickers`.

        Parameters
        ----------
        tickers: list
            List of ticker names
        freq: str
            Resampling frequency
        Returns
        -------
        df: pandas.DataFrame | pandas.Series
            Table of Adjusted Close prices for `tickers`
        """
        if isinstance(csv, str):
            return cls._csv(csv, tickers).loc[start_date:end_date]
        else:
            # tmp dictionary of panda.Series
            data = {}
            for i, ticker in enumerate(tickers):
                tmp_df = cls._get(
                    ticker, start_date=start_date, end_date=end_date)
                # successful data fetchinf
                if tmp_df is not None:
                    data[ticker] = tmp_df[cls._col]
            # dict to pandas.DataFrame
            df = pd.DataFrame(data)
        return df.sort_index(ascending=True).resample(freq).last()

    @classmethod
    def SP500(cls, return_prices_returns: bool = False, **kwargs):
        # fetch table of constituents
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                             header=0)[0]
        # keep columns of interest
        sp500 = sp500[['Ticker symbol', 'Security', 'GICS Sector']]
        # set ticker as index
        sp500.set_index('Ticker symbol', inplace=True)
        # fetch prices & returns
        if return_prices_returns:
            # get tickers list
            tickers = sp500.index.tolist()
            # pass arguments to method
            prices = cls.Prices(tickers, **kwargs)
            # calculate returns
            returns = prices.pct_change()[1:]
            return sp500, prices, returns
        # ignore prices & returns
        else:
            return sp500

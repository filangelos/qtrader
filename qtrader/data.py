# scientific computing
import numpy as np
import pandas as pd

# Quandl library
import os
import quandl as _quandl
_quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']


class _Transform(object):
    """Base class for transformations."""

    @classmethod
    def transform(cls, X):
        raise NotImplementedError


class Noise(_Transform):
    """Additive noise transformation."""

    @staticmethod
    def _func(X):
        return X + np.random.normal(0, 1, X.shape)

    @classmethod
    def transform(cls, X):
        return cls._func(X)


class Sinusoidal(_Transform):
    """Sinusoidal series transformation."""

    @staticmethod
    def _get_params(N):
        A = np.random.uniform(0, 1, N)
        W = np.random.uniform(0, 0.5, N)
        F = np.random.uniform(0, np.pi, N)
        return A, W, F

    @staticmethod
    def _func(A, w, f, X):
        return A * np.sin(2 * np.pi * w * X + f)

    @classmethod
    def transform(cls, X):
        _N = X.shape[0]
        return np.array(list(map(cls._func, *(cls._get_params(_N) + (X,)))))


class Pipeline(_Transform):
    """Composite transformation, linear pipeline of transformations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, X):
        _X = np.copy(X)
        for _transform in self.transforms:
            _X = _transform.transform(_X)
        return _X


class Quandl:
    """Quandl Wrapper."""
    start_date = None
    end_date = None

    @staticmethod
    def _get(ticker, **kwargs):
        """Helpder method for `quandl.get`.

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
        return _quandl.get('WIKI/%s' % ticker, **kwargs)

    @classmethod
    def Returns(cls, tickers):
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
        return pd.DataFrame.from_dict({ticker: cls._get(ticker, start_date=cls.start_date,
                                                        end_date=cls.end_date, transformation="rdiff")['Adj. Close'] for ticker in tickers})

    @classmethod
    def Prices(cls, tickers):
        """Get daily prices for `tickers`.

        Parameters
        ----------
        tickers: list
            List of ticker names

        Returns
        -------
        df: pandas.DataFrame
            Table of Adjusted Close prices for `tickers`
        """
        return pd.DataFrame.from_dict({ticker: cls._get(ticker, start_date=cls.start_date,
                                                        end_date=cls.end_date)['Adj. Close'] for ticker in tickers})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qtrader.utils.econometric import pnl as PnL


def time_series(series, title='', xlabel='', ylabel='', path=None):
    """Plot univariate and multivariate time series.

    Parameters
    ----------
    series : np.ndarray | pd.Series | pd.DataFrame
        Univariate / Multivariate time series to plot.
    title: str, optional
        Title for the figure.
    xlabel: str, optional
        Label for x-axis for the figure.
    ylabel: str, optional
        Label for y-axis for the figure.
    path: str, optional
        Path to store figure.
    """
    fig, ax = plt.subplots()
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        series.plot(ax=ax)
    elif isinstance(series, np.ndarray):
        if series.ndim == 1:
            plt.plot(series)
        elif series.ndim == 2:
            for c in range(series.shape[1]):
                plt.plot(series[:, c])
        else:
            raise ValueError('returns tensor cannot be handled')
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
    if path is not None:
        fig.savefig(path)
    fig.show()


def pnl(returns, path=None):
    """Plot profit and loss (PnL) from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
    path: str, optional
        Path to store figure.
    """
    _pnl = PnL(returns)
    if hasattr(returns, 'name'):
        title = f"{returns.name}: Profit & Loss"
    else:
        title = 'Profit & Loss'
    xlabel = 'Time'
    ylabel = 'Wealth Level'
    time_series(_pnl, title, xlabel, ylabel, path)


def trades(prices, weights, path=None):
    """Plot stock prices and corresponding portfolio weights.

    Parameters
    ----------
    prices : pandas.Series
        Asset prices.
    weights: pandas.Series
        Portfolio weights.
    path: str, optional
        Path to store figure.
    """
    fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
        'height_ratios': [4, 1], 'wspace': 0.01})
    prices.plot(ax=axes[0], color='b')
    weights.plot(ax=axes[1], color='g')
    fig.subplots_adjust(hspace=.0)
    if path is not None:
        fig.savefig(path)
    fig.show()


def table_image(array, path=None):
    """Plot 2D data as image.

    Parameters
    ----------
    array: numpy.ndarray | pandas.DataFrame
        2D data to be plotted.
    path: str, optional
        Path to store figure.
    """
    if array.ndim != 2:
        raise ValueError('array must be 2D')
    fig, ax = plt.subplots()
    ax.imshow(array, cmap=plt.cm.Greys)
    ax.axis('off')
    if path is not None:
        fig.savefig(path)
    fig.show()

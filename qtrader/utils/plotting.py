import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qtrader.utils.econometric import pnl as _PnL
from qtrader.utils.econometric import drawdown as _drawdown
from qtrader.utils.econometric import max_drawdown as _max_drawdown


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
    ax.xaxis.set_tick_params(rotation=45)
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
    _pnl = _PnL(returns)
    if hasattr(returns, 'name'):
        title = f"{returns.name or 'Strategy'}: Profit & Loss"
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
        'height_ratios': [3, 1], 'wspace': 0.01})
    axes[0].plot(prices.index, prices.values, color='b')
    axes[1].bar(weights.index, weights.values, color='g')
    axes[0].set(title='%s: Prices & Portfolio Weights' % (prices.name or 'Strategy'),
                ylabel='Price, $p_{t}$')
    axes[1].set(xlabel='Time', ylabel='Weight, $w_{t}$', ylim=[0, 1])
    axes[1].xaxis.set_tick_params(rotation=45)
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


def drawdown(returns, path=None):
    """Plot drawdown along with PnL.

    Prameters
    ---------
    returns: pandas.Series
        Returns of the strategy as a percentage, noncumulative.
    path: str, optional
        Path to store figure.
    """
    pnl = _PnL(returns)
    neg_drawdown = - _drawdown(returns)
    neg_max_drawdown = - _max_drawdown(returns)
    fig, ax = plt.subplots()
    pnl.plot(label='Profit & Loss', ax=ax)
    neg_drawdown.plot(label='Drawdown', ax=ax)
    neg_max_drawdown.plot(label='Max Drawdown', ax=ax)
    ax.set(title=f'{returns.name or "Strategy"}: Profit & Loss with Drawdown',
           ylabel='Wealth Level', xlabel='Time')
    ax.legend()
    if path is not None:
        fig.savefig(path)
    fig.show()

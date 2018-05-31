import numpy as np
import pandas as pd

from qtrader.utils.numpy import eps


def cum_returns(returns):
    """Computes cumulative returns from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    cumulative_returns : np.ndarray | pd.Series | pd.DataFrame
        Cumulative returns.
    """
    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)
    out = np.subtract(out, 1)
    return out


def pnl(returns):
    """Computes profit and loss (PnL) from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    pnl : np.ndarray | pd.Series | pd.DataFrame
        Profit and loss.
    """
    if returns.ndim > 2:
        raise ValueError('returns tensor cannot be handled')
    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)
    return out


def sharpe_ratio(returns):
    """Computes Sharpe Ratio from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    sharpe_ratio : float | np.ndarray | pd.Series
        Sharpe ratio.
    """
    if returns.ndim > 2:
        raise ValueError('returns tensor cannot be handled')
    return np.sqrt(len(returns)) * \
        np.mean(returns, axis=0) / (np.std(returns, axis=0) + eps)


def hit_ratio(returns):
    """Computes Hit Ratio from simple returns,
    represented by number of positive trades
    over total number of trades.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    hit_ratio : float | np.ndarray | pd.Series
        Hit ratio.
    """
    if returns.ndim > 2:
        raise ValueError('returns tensor cannot be handled')
    return np.sum(returns > 0, axis=0) / len(returns)


def awal(returns):
    """Computes Average Win to Average Loss ratio.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    awal : float | np.ndarray | pd.Series
        Average win to average loss ratio.
    """
    if returns.ndim > 2:
        raise ValueError('returns tensor cannot be handled')
    aw = returns[returns > 0].mean(axis=0)
    al = returns[returns < 0].mean(axis=0)
    return np.abs((aw+eps)/(al+eps))


def appt(returns):
    """Computes Average Profitability Per Trade.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    appt : float | np.ndarray | pd.Series
        Average profitability per trade.
    """
    if returns.ndim > 2:
        raise ValueError('returns tensor cannot be handled')
    pw = np.sum(returns > 0, axis=0) / len(returns)
    pl = np.sum(returns < 0, axis=0) / len(returns)
    aw = returns[returns > 0].mean(axis=0)
    al = returns[returns < 0].mean(axis=0)
    return pw * aw - pl * al


def drawdown(returns):
    """Computes Drawdown given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    drawdown : pandas.Series
        Drawdown of strategy.
    """
    _cum_returns = cum_returns(returns)
    expanding_max = _cum_returns.expanding(1).max()
    drawdown = expanding_max - _cum_returns
    return drawdown


def _drawdown(returns):
    """Computes Drawdown given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    drawdown : pandas.Series
        Drawdown of strategy.
    """
    _cum_returns = cum_returns(returns)
    drawdown = pd.Series(index=_cum_returns.index, name='drawdown')
    for T in _cum_returns.index[1:]:
        X_t = _cum_returns.loc[:T]
        X_T = _cum_returns.loc[T]
        drawdown.loc[T] = max([0, X_t.max() - X_T])
    return drawdown


def max_drawdown(returns):
    """Computes Max Drawdown given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    max_drawdown : pandas.Series
        Max drawdown of strategy.
    """
    _drawdown = drawdown(returns)
    return _drawdown.expanding(1).max()


def average_drawdown_time(returns):
    """Computes Average Drawdown Time given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    average_drawdown_time : datetime.timedelta
        Average drawdown time of strategy.
    """
    _drawdown = drawdown(returns)
    return _drawdown[_drawdown == 0].index.to_series().diff().mean()


def mean_returns(returns):
    """Compute mean returns given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    mean_returns : float
        Mean returns of strategy.
    """
    return returns.mean(axis=0)


def std_returns(returns):
    """Compute standard deviation of returns given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    std_returns : float
        Standard deviation of returns of strategy.
    """
    return returns.std(axis=0)


def skewness(returns):
    """Compute skewness of returns given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    skew_returns : float
        Skewness of returns of strategy.
    """
    return returns.skew(axis=0)


def kurtosis(returns):
    """Compute kurtosis of returns given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    kurt_returns : float
        Skewness of returns of strategy.
    """
    return returns.kurt(axis=0)


def tail_ratio(returns):
    """Compute tail ratio of returns given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    tail_ratio : float
        Tail ratio of returns of strategy.
    """
    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def value_at_risk(returns, cutoff=0.05):
    """Compute Value at risk (VaR) of a returns stream.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of returns.

    Returns
    -------
    VaR : float
        The VaR value.
    """
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(returns, cutoff=0.05):
    """Compute Conditional value at risk (CVaR) of a returns stream.
    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of returns.

    Returns
    -------
    CVaR : float
        The CVaR value.
    """
    # PERF: Instead of using the 'value_at_risk' function to find the cutoff
    # value, which requires a call to numpy.percentile, determine the cutoff
    # index manually and partition out the lowest returns values. The value at
    # the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])

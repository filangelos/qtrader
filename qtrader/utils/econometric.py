import numpy as np

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
    return np.abs(aw/(al+eps))


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

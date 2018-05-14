import numpy as np


def PnL(returns):
    """Profit and Loss given relative
    `returns` over time.

    Parameters
    ----------
    returns: iterable
        Relative returns (P_{t} / P_{t-1} - 1)
        in ascending time order

    Returns
    -------
    pnl: numpy.ndarray
        Wealth level over time
    """
    wealth = [1]
    for r in returns:
        wealth.append(wealth[-1] * (1+r))
    return np.array(wealth)

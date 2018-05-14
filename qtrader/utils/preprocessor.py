import numpy as np
import pandas as pd


def rolling1d(series, window):
    """Rolling window for a 1D series.

    Parameters
    ----------
    series: list | numpy.ndarray | pandas.Series
        Sequential 1D data
    window: int
        Window size

    Returns
    -------
    matrix: numpy.ndarray
        Matrix of rolling windowed series
    """
    if isinstance(series, list):
        series = np.array(series)
    if len(series.shape) != 1:
        raise ValueError("1D array expected")
    shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)


def rolling2d(array, window):
    """Rolling window for 2D array.

    Parameters
    ----------
    array: list | numpy.ndarray | pandas.DataFrame
        Sequential 2D data
    window: int
        Window size

    Returns
    -------
    matrix: numpy.ndarray
        Matrix of rolling windowed series
    """
    if isinstance(array, list):
        array = np.array(array)
    if len(array.shape) != 2:
        raise ValueError("2D array expected")
    out = np.empty((array.shape[0] - window + 1, window, array.shape[1]))
    if isinstance(array, np.ndarray):
        for i, col in enumerate(array.T):
            out[:, :, i] = rolling1d(col, window)
    elif isinstance(array, pd.DataFrame):
        for i, label in enumerate(array):
            out[:, :, i] = rolling1d(array[label], window)
    return out


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.

    References
    ----------
    .. [NC] Nolan Conaway, A softmax function for numpy,
            online available at
            https://nolanbconaway.github.io/blog/2017/softmax-numpy

    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()
    return p

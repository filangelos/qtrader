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

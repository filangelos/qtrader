import numpy as np
import pandas as pd

from qtrader.utils.numpy import eps


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


def Xy(series, window, out_shape=1):
    """Time series to supervised data.

    Parameters
    ----------
    series: list | numpy.ndarray | pandas.Series
        Sequential 1D data
    window: int
        Window size
    out_shape: int
        Output vectors shape

    Returns
    -------
    X: numpy.ndarray
        Feature matrix
    y: numpy.ndarray
        Target matrix
    """
    tmp = rolling1d(series, window+out_shape)
    X = tmp[:, :-out_shape]
    y = tmp[:, -out_shape:]
    return X, y


def standard(array):
    """Standardise data, column-wise.

    Parameters
    ----------
    array: numpy.ndarray | pandas.Series | pandas.DataFrame
        Data to be standardised.

    Returns
    -------
    standard_array: numpy.ndarray | pandas.Series | pandas.DataFrame
        Standardised data.
    """
    if array.ndim > 3:
        raise ValueError('array must be up to 3rd order tensor')
    axis = array.ndim - 2
    return (array - array.mean(axis=axis, keepdims=True)) / \
        (array.std(axis=axis, keepdims=True) + eps)


def flatten(array):
    """Flatten 3D array to 2D.

    Parameters
    ----------
    array: numpy.ndarray
        3D data to be flattened.

    Returns
    flat_array: numpy.ndarray
        2D flattened array.
    """
    if array.ndim != 3:
        raise ValueError('array must be 3D')
    N, L, M = array.shape
    return array.reshape(N, L*M)


def deflatten(array, window):
    """De-flatten 2D array to 3D, given window.

    Parameters
    ----------
    array: numpy.ndarray
        2D data to de-flatten.

    Returns
    -------
    deflat_array: numpy.ndarray
        3D deflattened data.
    """
    if array.ndim != 2:
        raise ValueError('array must be 2D')
    N, LM = array.shape
    if (LM / window) % 1 != 0:
        raise ValueError('invalid window size')
    return array.reshape(N, window, LM // window)

import datetime

import numpy as np
import pandas as pd


def sinewaves(A: np.ndarray,
              w: np.ndarray,
              co: np.ndarray,
              num_samples: int,
              tickers: list,
              freq: str):
    """Generate dummy sinewaves.

    Parameters
    ----------
    A: numpy.ndarray
        Amplitudes vector
    w: numpy.ndarray
        Circular frequencies vector
    co: numpy.ndarray
        Initial phase vector
    num_samples: int
        Number of samples
    tickers: list
        List of tickers
    freq: str
        Resampling frequency
    """
    # time index
    t = np.linspace(-2*np.pi, 2*np.pi, num_samples).reshape(-1, 1)
    # phi of sinosuid
    phi = np.dot(w, t.T) + co
    # sinewaves
    y = A * np.sin(phi)
    y = y.T
    # datetime indexes
    index = pd.date_range(end=datetime.date.today(),
                          freq=freq, periods=num_samples)
    # pandas compatible data
    df = pd.DataFrame(y, columns=tickers, index=index)
    return df

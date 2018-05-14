import numpy as np


def append_row(array, row):
    """Add `row` in `array`.

    Parameters
    ----------
    array: numpy.ndarray
        Array to modify
    row: numpy.ndarray
        Row to append to the array

    Returns
    -------
    array: numpy.ndarray
        Modified array
    """
    return np.r_[array, [row]]

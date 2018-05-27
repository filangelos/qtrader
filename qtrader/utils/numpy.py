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


# machine accuracy
eps = np.finfo(float).eps


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

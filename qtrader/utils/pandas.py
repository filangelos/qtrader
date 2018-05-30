import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `pandas.DataFrame` from
    missing entries.

    Parameters
    ----------
    df: pandas.DataFrame
        Table to be cleaned.
    Returns
    -------
    clean_df: pandas.DataFrame
        Cleaned table.
    """
    # remove infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    # drop NaN values
    return df.dropna()


def align(target, source):
    """Align indexes, columns and names
    of two `pandas` objects.

    Parameters
    ----------
    target: numpy.ndarray
        Object to be aligned.
    source: pandas.Series | pandas.DataFrame
        Template object used for alignment.

    Returns
    -------
    obj: pandas.Series | pandas.DataFrame
        Aligned pandas object.
    """
    if target.shape != source.shape:
        raise ValueError('mismatched shapes, impossible to align')
    if isinstance(source, pd.Series):
        obj = pd.Series(target, index=source.index,
                        name=source.name)
    elif isinstance(source, pd.DataFrame):
        obj = pd.DataFrame(target, index=source.index,
                           columns=source.columns)
    else:
        raise ValueError('unrecognised source type')
    return obj

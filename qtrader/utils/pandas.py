import numpy as np


def clean(df):
    # remove infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    # drop NaN values
    return df.dropna()

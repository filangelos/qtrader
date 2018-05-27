import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR as _VAR
import statsmodels.tsa.vector_ar.util as var_util


def VAR(df, max_order=15):
    """Vector Autoregressive Baseline Generator."""
    # VAR model
    if isinstance(df, pd.DataFrame):
        var = _VAR(df.values)
    elif isinstance(df, np.ndarray):
        var = _VAR(df)
    # fit model
    model = var.fit(maxlags=max_order, ic='aic')
    # simulation
    ts_gen = var_util.varsim(model.coefs, model.intercept,
                             model.sigma_u, steps=len(df.values))
    # Create pandas DataFrame
    df_gen = pd.DataFrame(ts_gen, columns=df.columns,
                          index=df.index[-len(ts_gen):])
    return df_gen

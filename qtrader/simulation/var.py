import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.api import VAR as _VAR


def VAR(df, max_order=15, return_model=False):
    """Vector Autoregressive Baseline Generator."""
    # VAR model
    var = _VAR(df)
    # fit model
    model = var.fit(maxlags=max_order, ic='aic')
    # simulation
    ts_gen = model.simulate_var(len(df.values))
    # Create pandas DataFrame
    df_gen = pd.DataFrame(ts_gen, columns=df.columns,
                          index=df.index[-len(ts_gen):])
    if return_model:
        return df_gen, model
    else:
        return df_gen

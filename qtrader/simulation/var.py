import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR as _VAR
import statsmodels.tsa.vector_ar.util as var_util


def VAR(ts, max_order=15):
    """Vector Autoregressive Baseline Generator."""
    # VAR model
    if isinstance(ts, pd.DataFrame):
        var = _VAR(ts.values)
    elif isinstance(ts, np.ndarray):
        var = _VAR(ts)
    # optimal order
    order = var.select_order(max_order)['aic']
    # fit model
    model = var.fit(order)
    # simulation
    ts_gen = var_util.varsim(model.coefs, model.intercept,
                             model.sigma_u, steps=len(ts.values))
    return ts_gen

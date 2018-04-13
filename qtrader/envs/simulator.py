import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR as _VAR
import statsmodels.tsa.vector_ar.util as var_util


def AAFT(ts, random=np.random.uniform, random_state=None):
    """Amplitude Adjusted Fourier Transform Baseline Generator."""
    # set random seed
    np.random.seed(random_state)
    # 2d time-series format
    _ts = ts.reshape(len(ts), -1)
    # Odd number of samples
    if len(_ts) % 2 != 0:
        _ts = _ts[1:, :]
    # Generated time-series
    ts_gen = np.empty_like(_ts)
    for i, tsi in enumerate(_ts.T):
        # Fourier Transaformation (real-valued signal)
        F_tsi = np.fft.rfft(tsi)
        # Randomization of Phase
        rv_phase = np.exp(random(0, np.pi, len(F_tsi)) * 1.0j)
        # Generation of new time-series
        F_tsi_new = F_tsi * rv_phase
        # Inverse Fourier Transformation
        ts_gen[:, i] = np.fft.irfft(F_tsi_new)
    return ts_gen


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

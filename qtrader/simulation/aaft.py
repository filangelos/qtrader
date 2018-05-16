import numpy as np
import pandas as pd


def AAFT(df, random=np.random.uniform, random_state=None):
    """Amplitude Adjusted Fourier Transform Baseline Generator."""
    # set random seed
    np.random.seed(random_state)
    # Operate on numpy.ndarray
    ts = df.values
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
    # Create pandas DataFrame
    df_gen = pd.DataFrame(ts_gen, columns=df.columns,
                          index=df.index[-len(ts_gen):])
    return df_gen

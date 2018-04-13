import numpy as np

import qtrader


def generator(num_samples, data, optimizer, window):
    # input data shape
    N, M = data.shape
    # generated dataset
    X = np.empty((num_samples, window, M), dtype=float)
    y = np.zeros((num_samples, M), dtype=float)
    # iterate over rolling windows
    for i, frame in enumerate(qtrader.utils.rolling2d(data, window)):
        try:
            # empirical mean estimate
            mu_r = frame.mean(axis=0)
            # empirical covariance estimate
            Sigma_r = data.cov()
            # initial random weights
            w0 = np.random.uniform(0, 1.0, M)
            w0 = w0 / np.sum(w0)
            # observation
            X[i, :, :] = frame
            # optimal portfolio vector
            y[i, :] = optimizer(mu_r, Sigma_r, w0)
        except BaseException:
            pass
    return X, y

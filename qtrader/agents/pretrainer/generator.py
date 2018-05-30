import numpy as np

import qtrader


def generator(num_samples, data, optimizer, window, short_sales=True):
    # input data shape
    N, M = data.shape
    # generated dataset
    X = np.empty((num_samples-window+1, window, M), dtype=float)
    y = np.zeros((num_samples-window+1, M), dtype=float)
    # iterate over rolling windows
    for i, frame in enumerate(qtrader.utils.preprocessor.rolling2d(data, window)):
        try:
            # empirical mean estimate
            mu_r = np.mean(frame, axis=0)
            # empirical covariance estimate
            Sigma_r = np.cov(data.T)
            # initial random weights
            w0 = np.random.uniform(0, 1.0, M)
            w0 = w0 / np.sum(w0)
            # observation
            X[i, :, :] = frame
            # optimal portfolio vector
            y[i, :] = optimizer(mu_r, Sigma_r, w0, short_sales)
        except BaseException as e:
            print("[i=%d]" % i, e)
    return X, y

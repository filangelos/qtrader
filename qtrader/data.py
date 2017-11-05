import numpy as np


class _Transform:

    @classmethod
    def transform(cls, X, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _func():
        raise NotImplementedError

    @staticmethod
    def _get_params(self):
        raise NotImplementedError


class Noise(_Transform):

    @staticmethod
    def _func(X):
        return X + np.random.normal(0, 1, X.shape)

    @classmethod
    def transform(cls, X):
        return cls._func(X)


class Sinusoidal(_Transform):

    @staticmethod
    def _get_params(N):
        A = np.random.uniform(0, 1, N)
        W = np.random.uniform(0, 0.5, N)
        F = np.random.uniform(0, np.pi, N)
        return A, W, F

    @staticmethod
    def _func(A, w, f, X):
        return A * np.sin(2 * np.pi * w * X + f)

    @classmethod
    def transform(cls, X):
        _N = X.shape[0]
        return np.asarray(list(map(cls._func, *(cls._get_params(_N) + (X,)))))


class Pipeline(_Transform):

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, X):
        _X = np.copy(X)
        for _transform in self.transforms:
            _X = _transform.transform(_X)
        return _X

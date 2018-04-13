import numpy as np
import scipy.optimize


def optimizer(J, *args):
    def _optimizer(mu: np.ndarray, Sigma: np.ndarray,
                   w0: np.ndarray, short_sales: bool = True) -> np.ndarray:
        # number of assets
        M = mu.shape[0]
        # equality constraint: budget
        con_budget = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }
        if short_sales:
            bounds = [(0, None) for _ in range(M)]
        else:
            bounds = [(None, None) for _ in range(M)]
        # execute
        results = scipy.optimize.minimize(
            J, w0, (mu, Sigma, w0, *args),
            constraints=(con_budget),
            bounds=bounds,
            method='SLSQP'
        )
        # handle errors
        if not results.success:
            raise BaseException(results.message)
        # optimal portfolio weights
        w = results.x
        return w
    return _optimizer

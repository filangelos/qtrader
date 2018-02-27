from collections import deque

import numpy as np
import scipy.optimize

from qtrader.agents._base import Agent


class TangentAgent(Agent):
    """Tangent Portfolio agent."""

    def __init__(self, action_space, window=10):
        self.action_space = action_space
        self.memory = deque(maxlen=window)

    def observe(self, observation):
        self.memory.append(observation.values)

    def act(self, observation, reward, done):
        # deque -> np.array, for easy math
        memory = np.array(self.memory)
        # number of assets
        M = len(observation)
        # expected returns vector
        mu = np.mean(memory, axis=0).reshape(M, 1)
        if len(self.memory) != self.memory.maxlen:
            sigma = np.eye(M)
        else:
            # empirical covariance matrix
            sigma = np.cov(memory.T)
        # tangency portfolio solver
        action, _ = self._tangent(mu, sigma)
        return action

    def _tangent(self, mu, sigma, short_sales=False):
        """Tangency Portfolio for Sharpe Ratio."""
        def sharpe_ratio(w):
            """Objective function, Sharpe Ratio."""
            return -np.dot(w.T, mu) / np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        # number of assets
        M = mu.shape[0]
        # initial guess
        w0 = np.ones(M) / M
        # accepted values for w
        bounds = [((-1 if short_sales else 0), 1) for _ in range(M)]
        # equality constraint: weights sum up to 1.0
        con_portfolio_vector = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
        }
        results = scipy.optimize.minimize(
            sharpe_ratio, w0,
            constraints=(con_portfolio_vector),
            bounds=bounds,
            method='SLSQP'
        )
        if not results.success:
            raise BaseException(results.message)
        w = results.x
        mu = np.dot(w.T, mu)
        var = np.dot(w.T, np.dot(sigma, w))
        return w, (mu, var)

from collections import deque

import numpy as np
import scipy.optimize

import qtrader.agents.pretrainer
from qtrader.agents.base import Agent


class QuadraticAgent(Agent):
    """Quadratic Programming agent."""

    _id = 'quadratic'

    def __init__(self, action_space, J, window=10, *args):
        self.optimizer = qtrader.agents.pretrainer.optimizer(self._J(J), *args)
        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()

    def observe(self, observation, action, reward, done, next_observation):
        self.memory.append(observation.values)

    def act(self, observation):
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
        # quadratic programming portfolio solver
        try:
            self.w = self.optimizer(mu, sigma, self.w)
        except BaseException:
            pass
        return self.w

    def _J(self, J):
        if J is "sharpe_ratio":
            return qtrader.agents.pretrainer.objectives.sharpe_ratio
        elif J is "risk_aversion":
            return qtrader.agents.pretrainer.objectives.risk_aversion

import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from qtrader.agents.base import Agent
from qtrader.utils.numpy import softmax


class VARAgent(Agent):
    """Model-based VAR agent,
    trained offline on a
    historic dataset."""

    _id = 'VAR'

    def __init__(self, df, max_order=15, policy='softmax'):
        # initialize VAR model
        self.model = VAR(df)
        # fit model
        self.model = self.model.fit(maxlags=max_order,
                                    ic='aic')
        # memory used to cache observations
        self.memory = pd.DataFrame(columns=df.columns)
        # policy
        self.policy = 'softmax'

    def observe(self, observation, action, reward, done, next_observation):
        self.memory.append([observation['returns'],
                            next_observation['returns']])

    def act(self, observation):
        _returns = observation['returns']
        if len(self.memory) == 0:
            # random sample
            _values = np.random.uniform(0, 1, self.model.coefs.shape[-1])
        else:
            # forecast one step returns
            _values = self.model.forecast(self.memory.dropna().values, 1)[0]
        # softmax policy
        if self.policy == 'softmax':
            # to pandas.Series
            _action = pd.Series(_values,
                                index=_returns.index,
                                name=_returns.name)
            return softmax(_action)
        # LONG best stock policy
        elif self.policy == 'best':
            # one-hot vector
            _action = np.zeros_like(_values).ravel()
            _action[np.argmax(_values)] = 1.0
            # to pandas.Series
            _action = pd.Series(_action,
                                index=_returns.index,
                                name=_returns.name)
            return _action

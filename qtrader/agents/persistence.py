import numpy as np
import pandas as pd

from qtrader.agents.base import Agent
from qtrader.utils.numpy import softmax


class PersistenceAgent(Agent):
    """Model-based **persistence** agent,
    acting based on last observation
    (i.e returns at t-1),
    using softmax function."""

    _id = 'persistence'

    def __init__(self):
        pass

    def act(self, observation):
        _returns = observation['returns']
        if _returns.isnull().any():
            # random sample
            _values = pd.Series(np.random.uniform(0, 1, len(_returns)),
                                index=_returns.index,
                                name=_returns.name)
        else:
            # one step look back
            _values = _returns
        return softmax(_values)

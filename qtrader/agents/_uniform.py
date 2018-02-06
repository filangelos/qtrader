from __future__ import absolute_import
from __future__ import division

import numpy as np

from qtrader.agents._base import Agent


class UniformAgent(Agent):
    """Uniform agent."""

    def __init__(self, action_space):
        self.N = action_space.shape[0]

    def act(self, observation, reward, done):
        return np.ones(self.N) / self.N

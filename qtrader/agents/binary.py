import random
import numpy as np

from qtrader.agents.base import Agent


class BinaryAgent(Agent):
    """Binary trading agent."""

    _id = 'binary'

    def __init__(self, action_space):
        self.N = action_space.shape[0]

    def act(self, observation, reward, done):
        b = random.randint(0, self.N - 1)
        c = np.zeros(self.N)
        c[b] = 1
        return c

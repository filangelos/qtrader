import numpy as np


class Agent:
    """`Agent` Interface/Class."""

    _id = 'base'

    def __init__(self, **kwargs):
        raise NotImplementedError

    def observe(self, observation):
        raise NotImplementedError

    def act(self, observation, reward, done):
        raise NotImplementedError

    @property
    def name(self):
        return self._id

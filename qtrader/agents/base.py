import numpy as np

from abc import abstractmethod


class Agent:
    """`Agent` Interface/Class."""

    _id = 'base'

    def __init__(self, **kwargs):
        raise NotImplementedError

    #######
    # API
    #######

    @property
    def name(self):
        return self._id

    def begin_episode(self, observation):
        pass

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, action, reward, done, next_observation):
        pass

    def end_episode(self):
        pass

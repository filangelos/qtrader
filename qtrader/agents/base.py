import numpy as np


class BaseAgent:
    """`BaseAgent` Interface/Class."""

    def __init__(self, **kwargs):
        raise NotImplementedError

    def observe(self, observation):
        raise NotImplementedError

    def act(self, observation, reward, done):
        raise NotImplementedError

from __future__ import absolute_import
from __future__ import division

import numpy as np

import gym
from gym import spaces

import qtrader


class PortfolioVector(gym.Space):
    """OpenAI Gym Spaces Portfolio Vector Struct."""

    def __init__(self, num_instruments):
        """Constructs a `PortfolioVector` object.

        Parameters
        ----------
        num_instruments: int
            Cardinality of universe
        """
        self.low = -np.ones(num_instruments, dtype=float)
        self.high = np.ones(num_instruments, dtype=float)

    def sample(self):
        """Draw random `PortfolioVector` sample."""
        cache = []
        for _ in range(self.shape[0] - 1):
            remainder = 1.0 - np.abs(cache).sum()
            cache.append(
                np.random.uniform(-remainder, remainder))
        cache.append(1.0 - np.abs(cache).sum())
        return np.array(cache)

    def contains(self, x):
        """Assert if `x` in space."""
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all() and (np.abs(x).sum() == 1.0)

    @property
    def shape(self):
        """Shape of `PortfolioVector` object."""
        return self.low.shape

    def __repr__(self):
        return "PortfolioVector" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class TradingEnv(gym.Env):
    """OpenAI Gym Trading Environment."""

    def __init__(self, universe, **kwargs):
        """Constructs a `TradingEnv` object.

        Parameters
        ----------
        universe: list
            List of instruments universe

        Attributes
        ----------
        universe: list
            List of instruments universe
        num_instruments: int
            Cardinality of universe
        action_space: gym.Space
            Agent's action space
        observation_space: gym.Space
            Agent's observation space
        data: pandas.DataFrame
            Historic data for `universe`
        """
        qtrader.utils.valid_type(universe, list)

        self.universe = universe
        qtrader.framework.logger.info("`universe`=%s" % (self.universe))

        self.num_instruments = len(self.universe)
        qtrader.framework.logger.info(
            "`num_instruments`=%s" % (self.num_instruments))

        self.action_space = PortfolioVector(self.num_instruments)
        qtrader.framework.logger.info(
            "`action_space`=%s" % (self.action_space))

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, (self.num_instruments))
        qtrader.framework.logger.info(
            "`observation_space`=%s" % (self.observation_space))

        self.data = self._get_data(**kwargs).dropna()
        qtrader.framework.logger.info("`data`=%s" % (self.data.head()))

        self._counter = 0

    @property
    def index(self):
        """Current index."""
        return self.data.index[self._counter]

    def _get_data(self, **kwargs):
        raise NotImplementedError

    def _step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : numpy.array
            Portfolio vector

        Returns
        -------
        observation, reward, episode_over, info: tuple
            * observation: object
                Observation of the environment
            * reward: float
                Reward received after this step
            * done: bool
                Flag for finished episode
            * info: dict
                Information about this step
        """
        if not self.action_space.contains(action):
            raise ValueError(
                'invalid `action` attempted: %s' % (action)
            )
        self._counter += 1
        observation = self.data.loc[self.index, :]
        reward = np.dot(observation.values, action)
        done = self.index == self.data.index[-1]
        info = {}
        return observation, reward, done, info

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: object
            The initial observation of the space.
        """
        self._counter = 0
        return self.data.loc[self.index, :]

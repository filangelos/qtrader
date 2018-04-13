import numpy as np
import pandas as pd

import gym
from gym import spaces

import qtrader


class BaseEnv(gym.Env):
    """OpenAI Gym Trading Environment."""

    def __init__(self, universe, trading_period='W', **kwargs):
        """Constructs a `BaseEnv` object.

        Parameters
        ----------
        universe: list
            List of instruments universe
        trading_period: str
            Trading period offset alias, http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

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
        self.universe = universe
        qtrader.framework.logger.info("`universe`=%s" % (self.universe))

        self.trading_period = trading_period
        qtrader.framework.logger.info(
            "`trading_period`=%s" % (self.trading_period))

        self.num_instruments = len(self.universe)
        qtrader.framework.logger.info(
            "`num_instruments`=%s" % (self.num_instruments))

        self.action_space = qtrader.envs.spaces.PortfolioVector(
            self.num_instruments)
        qtrader.framework.logger.info(
            "`action_space`=%s" % (self.action_space))

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            (self.num_instruments,),
                                            dtype=np.float32)
        qtrader.framework.logger.info(
            "`observation_space`=%s" % (self.observation_space))

        self.data = pd.DataFrame(self._get_data(**kwargs).dropna())
        qtrader.framework.logger.info("`data`=%s" % (self.data.head()))

        self._counter = 0

    @property
    def index(self):
        """Current index."""
        return self.data.index[self._counter]

    def _get_data(self, **kwargs):
        raise NotImplementedError

    def step(self, action):
        """The agent takes a step in the environment.

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

    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: object
            The initial observation of the space.
        """
        self._counter = 0
        return self.data.loc[self.index, :]

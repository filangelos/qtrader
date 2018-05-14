import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing

import gym
from gym import spaces

import qtrader
from qtrader.agents.base import Agent

from abc import abstractmethod


class BaseEnv(gym.Env):
    """OpenAI Gym Base Trading Environment.

    Attributes
    ----------
    universe: list
        List of instruments universe
    trading_period: str
        Trading period offset alias,
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    num_instruments: int
        Cardinality of universe
    action_space: gym.Space
        Agent's action space
    observation_space: gym.Space
        Agent's observation space
    prices: pandas.DataFrame
        Historic prices for `universe`
    returns: pandas.DataFrame
        Historic relative (percentage) returns for `universe`
    agents: list
        Registered agents that compete in the environment

    Methods
    -------
    step(action)
        The agent takes a step in the environment
    reset()
        Resets the state of the environment and returns an initial observation
    render()
        Present real-time data on a dashboard
    register(agent)
        Add an agent to the environment (stock market)
    """

    def __init__(self, universe, trading_period='W', **kwargs):
        self.universe: typing.List[str] = universe
        self.trading_period: str = trading_period
        # risky assets & cash under consideration
        self.num_instruments: int = len(self.universe) + 1
        # risky assets & cash portfolio vector
        self.action_space = qtrader.envs.spaces.PortfolioVector(
            self.num_instruments)
        # risky assets & cash prices vector
        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            (self.num_instruments,),
                                            dtype=np.float32)
        # market prices data
        self.prices = self._get_prices(**kwargs).dropna()
        # add cash column
        self.prices['CASH'] = 1.0
        # relative (percentage) returns
        self.returns = self.prices.pct_change()
        # counter to follow time index
        self._counter = 0
        # list of registered agents
        self.agents = []
        # agent's initial wealth
        self._pnl = pd.DataFrame(index=self.prices.index, columns=[
                                 agent.name for agent in self.agents])
        # figure & axes placeholders
        self._fig, self._axes = None, None

    @property
    def index(self) -> pd.DatetimeIndex:
        """Current index."""
        return self.prices.index[self._counter]

    @abstractmethod
    def _get_prices(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _get_observation(self) -> pd.DataFrame:
        return self.prices.loc[self.index, :]

    def _get_reward(self, action) -> float:
        return np.dot(self.returns.loc[self.index].values, action)

    def _get_done(self) -> bool:
        return self.index == self.prices.index[-1]

    def _get_info(self) -> dict:
        return {}

    def register(self, agent: Agent):
        """Register an `agent` to the environment."""
        if agent not in self.agents:
            self.agents.append(agent)
            self._pnl[agent.name] = np.nan
            self._pnl[agent.name].iloc[0] = 1.0
            qtrader.framework.logger.info(
                "New agent %s registered in %s" % (agent, self))
        else:
            qtrader.framework.logger.info(
                "%s already registered in %s" % (agent, self))

    def unregister(self, agent: Agent):
        """Unregister an `agent` from the environment."""
        if agent in self.agents:
            self.agents.remove(agent)
            self._pnl.drop(columns=[agent.name], inplace=True)

    def step(self, action) -> typing.Tuple[object, float, bool, dict]:
        """The agent takes a step in the environment.

        Parameters
        ----------
        action: numpy.array | list
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
        # check agents' availability
        if len(self.agents) == 0:
            raise RuntimeError('no agent registed in the environment')
        # timestep
        self._counter += 1
        # fetch return values
        observation = self._get_observation()
        done = self._get_done()
        info = self._get_info()
        # multiple registered agents
        if len(self.agents) > 1:
            reward = []
            # iterate over agents
            if len(action) != len(self.agents):
                raise ValueError(
                    'invalid number of actions provided'
                )
            for i, A in enumerate(action):
                # action validity check
                if not self.action_space.contains(A):
                    raise ValueError(
                        'invalid `action` attempted: %s' % (A)
                    )
                reward.append(self._get_reward(A))
                # calculate new wealth level
                self._pnl[self.agents[i].name].iloc[self._counter] = (
                    1+reward[-1]) * self._pnl[self.agents[i].name].iloc[self._counter - 1]
        # single agent
        else:
            # action validity check
            if not self.action_space.contains(action):
                raise ValueError(
                    'invalid `action` attempted: %s' % (action)
                )
            # calculate reward
            reward = self._get_reward(action)
            # calculate new wealth level
            self._pnl[self.agents[0].name].iloc[self._counter] = (
                1+reward) * self._pnl[self.agents[0].name].iloc[self._counter - 1]
        return observation, reward, done, info

    def reset(self) -> object:
        """Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: object
            The initial observation of the space.
        """
        # set time to zero
        self._counter = 0
        # reset agent's wealth
        self._pnl = pd.DataFrame(index=self.prices.index, columns=[
                                 agent.name for agent in self.agents])
        self._pnl.iloc[0] = [1.0 for _ in self.agents]
        return self.prices.loc[self.index, :]

    def render(self) -> None:
        """Graphical interface of environment."""
        # initialize figure and axes
        if self._fig is None or self._axes is None:
            # figure & axes for render()
            self._fig, self._axes = plt.subplots(ncols=2, figsize=(12.8, 4.8))
        # remove everything from the axes
        self._axes[0].clear()
        self._axes[1].clear()
        # axes content
        self.prices.loc[:self.index].plot(ax=self._axes[0])
        self._pnl.loc[:self.index].plot(ax=self._axes[1])
        # axes settings
        self._axes[0].set_xlim(self.prices.index.min(),
                               self.prices.index.max())
        self._axes[0].set_title('Market Prices')
        self._axes[0].set_ylabel('Prices')
        self._axes[1].set_xlim(self._pnl.index.min(),
                               self._pnl.index.max())
        self._axes[1].set_title('PnL')
        self._axes[1].set_ylabel('Wealth Level')
        # draw throttled
        plt.pause(0.0001)
        self._fig.canvas.draw()

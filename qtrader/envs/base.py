import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing

import gym

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
    action_space: qtrader.envs.spaces.PortfolioVector
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

    class Record:
        """Local data structure for actions and rewards records.

        Attributes
        ----------
        actions: pandas.DataFrame
            Table of actions performed by agent
        rewards: pandas.DataFrame
            Table of rewards received by agent
        """

        def __init__(self, index, columns):
            # records of actions
            self.actions = pd.DataFrame(
                columns=columns, index=index, dtype=float)
            self.actions.iloc[0] = np.zeros(len(columns))
            self.actions.iloc[0]['CASH'] = 1.0
            # records of rewards
            self.rewards = pd.DataFrame(
                columns=columns, index=index, dtype=float)
            self.rewards.iloc[0] = np.zeros(len(columns))

    def __init__(self,
                 universe: typing.Optional[typing.List[str]] = None,
                 prices: typing.Optional[pd.DataFrame] = None,
                 trading_period: str = 'W-FRI',
                 **kwargs):
        # --------------------------------------------------------------------------
        # either `universe` or `prices` non-None
        if not (universe is None) ^ (prices is None):
            raise ValueError(
                'either `universe` or `prices` should be non-None')
        # --------------------------------------------------------------------------
        # set trading period
        self.trading_period = trading_period
        # <prices> provided
        if prices is not None and isinstance(prices, pd.DataFrame):
            # prices table
            self._prices = qtrader.utils.pandas.clean(
                prices.resample(self.trading_period).last())
        # <universe> provided
        elif universe is not None and isinstance(universe, list):
            # fetch prices
            self._prices = qtrader.utils.pandas.clean(
                self._get_prices(universe,
                                 trading_period=self.trading_period, **kwargs))
        # --------------------------------------------------------------------------
        # risky assets & cash under consideration
        num_instruments: int = len(self.universe) + 1
        # risky assets & cash portfolio vector
        self.action_space = qtrader.envs.spaces.PortfolioVector(
            num_instruments)
        # risky assets & cash prices vector
        self.observation_space = gym.spaces.Box(-np.inf,
                                                np.inf,
                                                (num_instruments,),
                                                dtype=np.float32)
        # --------------------------------------------------------------------------
        # add cash column
        self._prices['CASH'] = 1.0
        # relative (percentage) returns
        self._returns = self._prices.pct_change()
        # --------------------------------------------------------------------------
        # counter to follow time index
        self._counter = 0
        # --------------------------------------------------------------------------
        # dictionary of registered agents
        self.agents = {}
        # agent's initial wealth
        self._pnl = pd.DataFrame(index=self.dates, columns=[
                                 agent.name for agent in self.agents])
        # --------------------------------------------------------------------------
        # figure & axes placeholders
        self._fig, self._axes = None, None

    @property
    def universe(self):
        """List of instruments universe."""
        return self._prices.columns.tolist()

    @property
    def dates(self):
        """Dates of the environment prices."""
        return self._prices.index

    @property
    def index(self) -> pd.DatetimeIndex:
        """Current index."""
        return self.dates[self._counter]

    @property
    def _max_episode_steps(self) -> int:
        """Number of timesteps available."""
        return len(self.dates)

    @abstractmethod
    def _get_prices(self, universe, trading_period, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _get_observation(self) -> object:
        ob = {}
        ob['prices'] = self._prices.loc[self.index, :]
        ob['returns'] = self._returns.loc[self.index, :]
        return ob

    def _get_reward(self, action) -> pd.Series:
        return self._returns.loc[self.index] * action

    def _get_done(self) -> bool:
        return self.index == self.dates[-1]

    def _get_info(self) -> dict:
        return {}

    def _validate_agents(self):
        """Check agents' availability."""
        if len(self.agents) == 0:
            raise RuntimeError('no agent registed in the environment')

    #######
    # API
    #######

    def register(self, agent: Agent):
        """Register an `agent` to the environment."""
        # verify interface
        if not hasattr(agent, 'name'):
            raise ValueError('agent must have a `name` attribute.')
        # verify uniqueness
        if agent.name not in self.agents:
            self.agents[agent.name] = self.Record(
                columns=self.universe, index=self.dates)

    def unregister(self, agent: typing.Optional[Agent]):
        """Unregister an `agent` from the environment."""
        # when agent=None, unregister all agents
        if agent is None:
            # clean records
            self.agents = {}
            return None
        # --------------------------------------------------------------------------
        # verify interface
        if not hasattr(agent, 'name'):
            raise ValueError('agent must have a `name` attribute.')
        # verify availability
        if agent.name in self.agents:
            del self.agents[agent.name]

    def step(self, action: typing.Union[object, typing.Dict[str, object]]):
        """The agent takes a step in the environment.

        Parameters
        ----------
        action: numpy.array | dict
            Portfolio vector(s)

        Returns
        -------
        observation, reward, episode_over, info: tuple
            * observation: object
                Observation of the environment
            * reward: float | dict
                Reward(s) received after this step
            * done: bool
                Flag for finished episode
            * info: dict
                Information about this step
        """
        self._validate_agents()
        # timestep
        self._counter += 1
        # fetch return values
        observation = self._get_observation()
        done = self._get_done()
        info = self._get_info()
        # verify interface
        if action.keys() != self.agents.keys():
            raise ValueError(
                'invalid interface of actions provided'
            )
        # container
        reward = {}
        # iterate over agents
        for name, A in action.items():
            # action validity check
            if not self.action_space.contains(A):
                raise ValueError(
                    'invalid `action` attempted: %s' % (A)
                )
            # actions buffer
            self.agents[name].actions.loc[self.index] = A
            self.agents[name].rewards.loc[self.index] = self._get_reward(A)
            # return value
            reward[name] = self.agents[name].rewards.loc[self.index].sum()
        return observation, reward, done, info

    def reset(self) -> object:
        """Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: object
            The initial observation of the space.
        """
        self._validate_agents()
        # set time to zero
        self._counter = 0
        # get initial observation
        ob = self._get_observation()
        return ob

    def render(self) -> None:
        """Graphical interface of environment."""
        # initialize figure and axes
        if self._fig is None or self._axes is None:
            # figure & axes for render()
            self._fig, self._axes = plt.subplots(ncols=2, figsize=(12.8, 4.8))
        #
        _pnl = pd.DataFrame(columns=self.agents.keys(),
                            index=self.dates)
        # calculate PnL
        for agent in self.agents:
            # collapse date-wise rewards
            _pnl[agent] = (self.agents[agent].rewards.sum(
                axis=1) + 1).cumprod()
        # remove everything from the axes
        self._axes[0].clear()
        self._axes[1].clear()
        # axes content
        self._prices.loc[:self.index].plot(ax=self._axes[0])
        _pnl.loc[:self.index].plot(ax=self._axes[1])
        # axes settings
        self._axes[0].set_xlim(self.dates.min(),
                               self.dates.max())
        self._axes[0].set_title('Market Prices')
        self._axes[0].set_ylabel('Prices')
        self._axes[1].set_xlim(self._pnl.index.min(),
                               self._pnl.index.max())
        self._axes[1].set_title('PnL')
        self._axes[1].set_ylabel('Wealth Level')
        # draw throttled
        plt.pause(0.0001)
        self._fig.canvas.draw()

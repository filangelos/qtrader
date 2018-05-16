import qtrader
from qtrader.simulation.tests.base import Test

import numpy as np

import matplotlib.pyplot as plt


class Arbitrage(Test):

    @classmethod
    def run(cls, df_1, df_2, tickers, freq='b', window=20, render=False):
        cls._test_env(df_1, tickers, freq, window, render)
        cls._test_env(df_2, tickers, freq, window, render)
        return False

    @classmethod
    def _test_env(cls, df, tickers, freq, window, render):
        # initial prices
        po = 500
        # prices as cumulative product of returns
        prices = po*np.cumprod(df+1)
        # create environment out of family 1 data
        env = qtrader.envs.TradingEnv(
            tickers, trading_period=freq, prices=prices)
        # list of agents
        tangent_agent = qtrader.agents.QuadraticAgent(
            env.action_space, 'sharpe_ratio', window, 0.0)
        random_agent = qtrader.agents.RandomAgent(env.action_space)
        # register agents to environment
        env.register(tangent_agent)
        env.register(random_agent)
        # training loop
        ob = env.reset()
        tangent_reward = 0.0
        random_reward = 0.0
        done = False
        info = {}
        while not done:
            # observations
            tangent_agent.observe(ob)
            # action calculations
            tangent_action = tangent_agent.act(ob, tangent_reward, done)
            random_action = random_agent.act(ob, random_reward, done)
            # take actions
            ob, (tangent_reward, random_reward), done, info = env.step(
                [tangent_action, random_action])
        # plot comparison of wealth levels
        if render:
            # initialize figure and axes
            fig, ax = plt.subplots()
            # plot PnLs
            env._pnl.plot(ax=ax)
            ax.set_title('Agents\' PnL')
            ax.set_ylabel('Wealth Level')

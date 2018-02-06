from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import qtrader

import gym

CSV_PATH = 'tests/tmp/data/prices.csv'


class TestAgents(unittest.TestCase):
    """Test `qtrader.agents` module."""

    def test__RandomAgent(self):
        """Test `qtrader.agents.RandomAgent` class."""
        env = gym.make('FrozenLake-v0')
        ob = env.reset()
        reward = 0
        done = False
        agent = qtrader.agents.RandomAgent(env.action_space)
        for _ in range(10):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
        return self.assertTrue(True)

    def test__UniformAgent(self):
        """Test `qtrader.agents.UniformAgent` class."""
        env = qtrader.envs.DailyReturnEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        ob = env.reset()
        reward = 0
        done = False
        agent = qtrader.agents.UniformAgent(env.action_space)
        for _ in range(10):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
        return self.assertTrue(True)

    def test__TangentAgent(self):
        """Test `qtrader.agents.TangentAgent` class."""
        env = qtrader.envs.DailyReturnEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        # play with tangent portfolio agent
        ob = env.reset()
        reward = 0
        done = False
        tangent_agent = qtrader.agents.TangentAgent(env.action_space)
        tangent_cumsum = 0
        while not done:
            tangent_agent.observe(ob)
            action = tangent_agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            tangent_cumsum += reward
        # play with random agent
        ob = env.reset()
        reward = 0
        done = False
        random_agent = qtrader.agents.RandomAgent(env.action_space)
        random_cumsum = 0
        while not done:
            action = random_agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            random_cumsum += reward
        # expect tangent to outperform random
        return tangent_cumsum > random_cumsum


if __name__ == '__main__':
    unittest.main()

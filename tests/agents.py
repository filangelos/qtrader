import unittest

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
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        ob = env.reset()
        reward = 0
        done = False
        agent = qtrader.agents.UniformAgent(env.action_space)
        for _ in range(10):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
        return self.assertTrue(True)

    def test__QuadraticAgent_Tangent(self):
        """Test `qtrader.agents.QuadraticAgent` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        # play with tangent portfolio agent
        ob = env.reset()
        reward = 0
        done = False
        tangent_agent = qtrader.agents.QuadraticAgent(
            env.action_space, 'sharpe_ratio', 10, 0.5)
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

    def test__QuadraticAgent_Risk_Aversion(self):
        """Test `qtrader.agents.QuadraticAgent` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        # play with tangent portfolio agent
        ob = env.reset()
        reward = 0
        done = False
        risk_aversion_agent = qtrader.agents.QuadraticAgent(
            env.action_space, 'risk_aversion', 10, 0.1, 0.0025)
        risk_aversion_cumsum = 0
        while not done:
            risk_aversion_agent.observe(ob)
            action = risk_aversion_agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            risk_aversion_cumsum += reward
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
        # expect risk aversion to outperform random
        return risk_aversion_cumsum > random_cumsum


if __name__ == '__main__':
    unittest.main()

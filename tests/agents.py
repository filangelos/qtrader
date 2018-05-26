import unittest

import qtrader

import gym

CSV_PATH = 'db/prices.csv'


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
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
        return self.assertTrue(True)

    def test__UniformAgent(self):
        """Test `qtrader.agents.UniformAgent` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'JPM'], csv=CSV_PATH, end_date='2018')
        agent = qtrader.agents.UniformAgent(env.action_space)
        env.register(agent)
        ob = env.reset()
        reward = 0
        done = False
        for _ in range(10):
            action = agent.act(ob)
            ob, reward, done, _ = env.step({agent.name: action})
        env.unregister(agent)
        return self.assertTrue(True)

    def test__QuadraticAgent_Tangent(self):
        """Test `qtrader.agents.QuadraticAgent` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'JPM'], csv=CSV_PATH, end_date='2018')
        # play with tangent portfolio agent
        tangent_agent = qtrader.agents.QuadraticAgent(
            env.action_space, 'sharpe_ratio', 10, 0.5)
        env.register(tangent_agent)
        ob = env.reset()
        ob = ob['prices']
        reward = 0
        done = False
        tangent_cumsum = 0
        while not done:
            tangent_agent.observe(ob, None, reward, done, None)
            action = tangent_agent.act(ob)
            ob, reward, done, _ = env.step({tangent_agent.name: action})
            ob = ob['prices']
            tangent_cumsum += reward[tangent_agent.name]
        env.unregister(tangent_agent)
        # play with random agent
        random_agent = qtrader.agents.RandomAgent(env.action_space)
        env.register(random_agent)
        ob = env.reset()
        reward = 0
        done = False
        random_cumsum = 0
        while not done:
            action = random_agent.act(ob)
            ob, reward, done, _ = env.step({random_agent.name: action})
            random_cumsum += reward[random_agent.name]
        env.unregister(random_agent)
        # expect tangent to outperform random
        return tangent_cumsum > random_cumsum

    def test__QuadraticAgent_Risk_Aversion(self):
        """Test `qtrader.agents.QuadraticAgent` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'JPM'], csv=CSV_PATH, end_date='2018')
        # play with tangent portfolio agent
        risk_aversion_agent = qtrader.agents.QuadraticAgent(
            env.action_space, 'risk_aversion', 10, 0.1, 0.0025)
        env.register(risk_aversion_agent)
        ob = env.reset()
        ob = ob['prices']
        reward = 0
        done = False
        risk_aversion_cumsum = 0
        while not done:
            risk_aversion_agent.observe(ob, None, reward, done, None)
            action = risk_aversion_agent.act(ob)
            ob, reward, done, _ = env.step({risk_aversion_agent.name: action})
            ob = ob['prices']
            risk_aversion_cumsum += reward[risk_aversion_agent.name]
        env.unregister(risk_aversion_agent)
        # play with random agent
        random_agent = qtrader.agents.RandomAgent(env.action_space)
        env.register(random_agent)
        ob = env.reset()
        reward = 0
        done = False
        random_cumsum = 0
        while not done:
            action = random_agent.act(ob)
            ob, reward, done, _ = env.step({random_agent.name: action})
            random_cumsum += reward[random_agent.name]
        env.register(random_agent)
        # expect risk aversion to outperform random
        return risk_aversion_cumsum > random_cumsum


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import qtrader

CSV_PATH = 'db/prices.csv'


class TestEnvs(unittest.TestCase):
    """Test `qtrader.envs` module."""

    def test__TradingEnv(self):
        """Test `qtrader.envs.TradingEnv` class."""
        env = qtrader.envs.TradingEnv(
            ['AAPL', 'MSFT', 'GE', 'JPM'], csv=CSV_PATH, end_date='2018')
        env.reset()
        done = False
        rewards = []
        np.random.seed(13)
        agent = qtrader.agents.RandomAgent(env.action_space)
        env.register(agent)
        while not done:
            _, reward, done, _ = env.step(
                env.action_space.sample())  # random agent
            rewards.append(reward)
        env.unregister(agent)
        return self.assertIsInstance(np.sum(rewards), float)


if __name__ == '__main__':
    unittest.main()

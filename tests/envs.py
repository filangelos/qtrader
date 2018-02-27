import unittest

import numpy as np
import qtrader

CSV_PATH = 'tests/tmp/data/prices.csv'


class TestEnvs(unittest.TestCase):
    """Test `qtrader.envs` module."""

    def test__DailyReturnEnv(self):
        """Test `qtrader.envs.DailyReturnEnv` class."""
        env = qtrader.envs.DailyReturnEnv(
            ['AAPL', 'MSFT', 'GE', 'VOD'], csv=CSV_PATH, end_date='2018')
        env.reset()
        done = False
        rewards = []
        np.random.seed(13)
        while not done:
            _, reward, done, _ = env.step(
                env.action_space.sample())  # random agent
            rewards.append(reward)
        return self.assertIsInstance(np.sum(rewards), float)


if __name__ == '__main__':
    unittest.main()

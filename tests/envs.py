from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import qtrader


class TestEnvs(unittest.TestCase):
    """Test `qtrader.envs` module."""

    def test__DailyReturnEnv(self):
        """Test `qtrader.envs.DailyReturnEnv` class."""
        env = qtrader.envs.DailyReturnEnv(
            ['WIKI/AAPL', 'WIKI/MSFT', 'WIKI/GOOGL'], source='quandl', start_date='2016-01-01')
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import qtrader

import gym


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


if __name__ == '__main__':
    unittest.main()

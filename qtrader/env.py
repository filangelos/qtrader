import gym
import numpy as np
from gym import spaces

import logging
import os


class TradingEnv(gym.Env):
    """OpenAI Gym Trading Environment."""

    def __init__(self, universe):
        """Constrcuts `TradingEnv` object.

        Parameters
        ----------
        universe: list
            List of instruments universe

        Attributes
        ----------
        universe: list
            List of instruments universe
        num_instruments: int
            Cardinality of universe
        logger: logging.Logger
            Logging handler
        """
        logger = logging.getLogger('qtrader')

        if not isinstance(universe, list):
            raise TypeError(
                'invalid `universe` type; "%s" != "list"' % type(universe))

        self.universe = universe
        logger.info("`universe`=%s" % self.universe)

        self.num_instruments = len(self.universe)
        logger.info("`num_instruments`=%s" % self.universe)

    def _step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : numpy.array
            Portfolio vector

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        return ob, reward, episode_over, info

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        return self._get_state()

    def _render(self, mode='human', close=False):
        pass

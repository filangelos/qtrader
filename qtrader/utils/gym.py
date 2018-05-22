import numpy as np

from gym import Env
from qtrader.agents.base import Agent

import typing


def cardinalities(env: Env) -> typing.Tuple[int, int]:
    """Fetch observation and action spaces cardinalities.

    Parameters
    ----------
    env: gym.Env
        OpenAI Gym compatible environment

    Returns
    -------
    observation_space_cardinality: int
        Cardinality of observation space
    action_space_cardinality: int
        Cardinality of action space
    """
    if hasattr(env.observation_space, 'n'):
        observation_space_cardinality = env.observation_space.n
    else:
        observation_space_cardinality = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        action_space_cardinality = env.action_space.n
    else:
        action_space_cardinality = env.action_space.shape[0]
    return observation_space_cardinality, action_space_cardinality


def run(env: Env, agent: Agent) -> float:
    """Run episode on the `env` using `agent`.

    Parameters
    ----------
    env: gym.Env
        OpenAI Gym compatible environment
    agent: qtrader.agent.base.Agent
        Agent to interact with the environment

    Returns
    -------
    cumulative_reward: float
        Cumulative reward of one episode
    """
    # initialize cumulative reward
    cumulative_reward = 0.0
    # environment: reset & fetch observation
    ob = env.reset()
    # initialize reward
    reward = 0.0
    # termination flag
    done = False
    # environment state information
    info = {}
    # iterator for maximum episode steps
    j = 0
    # agent closure: beginning of episode
    agent.begin_episode(ob)
    # interaction loop
    while (not done) and (j < env._max_episode_steps):
        # agent closure: determine action
        action = agent.act(ob)
        # environment: take action
        ob_, reward, done, info = env.step(action)
        # increment cumulative reward
        cumulative_reward = cumulative_reward + reward
        # agent closure: observe
        agent.observe(ob, action, reward, done, ob_)
        # set new observation to current
        ob = ob_
        # increment iterator
        j = j + 1
    # agent closure: end of episode
    agent.end_episode()
    return cumulative_reward


def one_hot(discrete, action_space_cardinality):
    action = np.zeros((action_space_cardinality), dtype=int)
    action[discrete] = 1
    return action

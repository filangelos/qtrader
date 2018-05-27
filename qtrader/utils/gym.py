import numpy as np

from gym import Env

import typing
import os


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


def run(env: Env, agent,
        num_episodes: int,
        record: bool = True, log: bool = False):
    """Run episode on the `env` using `agent`.

    Parameters
    ----------
    env: gym.Env
        OpenAI Gym compatible environment
    agent: qtrader.agent.base.Agent
        Agent to interact with the environment
    num_episodes: int
        Number of episodes to run
    record: bool
        Keep record of actions & rewards for each step
    log: bool
        Flag for logging at the end of each episode

    Returns
    -------
    rewards: list
        List of rewards per step per episode
    actions: list
        List of actions per step per episode
    """
    # unregister all agents from environment
    if hasattr(env, 'unregister'):
        # when agent=None, all agents unregistered
        env.unregister(agent=None)
    # register agent to environment if needed
    if hasattr(env, 'register'):
        # assign random name if not given
        if not hasattr(agent, 'name'):
            agent.name = '_default'
        # register agent, duplicates are ignored
        env.register(agent)
    # initialize rewards buffer
    rewards = []
    # initialize actions buffer
    actions = []
    # best reward for saving model
    _best_reward = -np.inf

    def _run() -> typing.Tuple[typing.List[float], typing.List[np.ndarray]]:
        """Closure runner for each episode."""
        # initialize rewards local buffer
        _rewards = []
        # initialize actions local buffer
        _actions = []
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
            # class 1: trading environments
            if hasattr(env, 'register'):
                # environment: take action
                ob_, reward, done, info = env.step({agent.name: action})
                # fetch agent-specific reward
                reward = reward[agent.name]
            # class 2: vanilla environments
            else:
                # environment: take action
                ob_, reward, done, info = env.step(action)
            # store reward
            _rewards.append(reward)
            # store action
            _actions.append(action)
            # agent closure: observe
            agent.observe(ob, action, reward, done, ob_)
            # set new observation to current
            ob = ob_
            # increment iterator
            j = j + 1
        # agent closure: end of episode
        agent.end_episode()
        return _rewards, _actions

    for e in range(num_episodes):
        # run episode
        R, A = _run()
        # episode-wise records
        if record:
            # store rewards
            rewards.append(R)
            # store actions
            actions.append(A)
        # log cumulative rewards
        if log:
            print('episode: %4d, cumulative reward: %+.5f' % (e, sum(R)))
        if sum(R) > _best_reward:
            # try to delete previously best model
            try:
                os.remove('tmp/models/%s/%f.h5' % (agent.name, _best_reward))
            except:
                pass
            # set best reward
            _best_reward = sum(R)
            # store agent state
            if hasattr(agent, 'save'):
                # create folder if not there
                if not os.path.exists('tmp/models/%s' % agent.name):
                    os.makedirs('tmp/models/%s' % agent.name)
                # store weights
                agent.save('tmp/models/%s/%f.h5' % (agent.name, _best_reward))

    return rewards, actions


def one_hot(discrete, action_space_cardinality):
    action = np.zeros((action_space_cardinality), dtype=int)
    action[discrete] = 1
    return action

import numpy as np
import pandas as pd

import tensorflow as tf

from qtrader.agents.base import Agent
from qtrader.utils.numpy import softmax
from qtrader.utils import rolling2d


class RNNAgent(Agent):
    """Model-based RNN agent,
    trained offline on a
    historic dataset."""

    _id = 'RNN-GRU'

    def __init__(self, df, hidden_units=64, policy='softmax',
                 batch_size=32, epochs=50):
        # get dimensions
        observation_size = action_size = len(df.columns)
        # initialize model
        self.model = self.build_model(observation_size,
                                      action_size,
                                      hidden_units)
        # prepare data
        X, y = self.Xy(df, window=1)
        # fit model
        self.model.fit(X, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=0)
        # memory used to cache observations
        self.memory = pd.DataFrame(columns=df.columns)
        # polict
        self.policy = policy

    def build_model(self,
                    observation_size,
                    action_size,
                    hidden_units):
        # input node
        X = tf.keras.layers.Input(
            shape=(1, observation_size),
            name='X'
        )
        # recurrent layer
        rnn = tf.keras.layers.GRU(
            hidden_units,
            activation='relu',
            name='rnn'
        )(X)
        # fully-connected layer
        y = tf.keras.layers.Dense(
            action_size,
            name='y'
        )(rnn)
        # model
        model = tf.keras.models.Model(X, y)
        # compilation
        model.compile(optimizer='rmsprop',
                      loss='mse')
        # print summary
        model.summary()
        return model

    def Xy(self, df, window):
        # rolling data
        tmp = rolling2d(clean(df), window + 1)
        # feature matrix
        X = tmp[:, :-1, :]
        # target matrix
        y = tmp[:, -1, :]
        return X, y

    def act(self, observation):
        # fetch historic returns
        _returns = observation['returns']
        # sth went wrong
        if np.isnan(np.sum(_returns)):
            # random sample
            _values = np.random.uniform(
                0, 1, self.model.input.shape.as_list()[-1])
        # normal case
        else:
            # forecast one step ahead
            _values = self.model.predict(_returns.values.reshape(1, 1, -1))[0]
        # std went wrong
        if np.isnan(np.sum(_values)):
            # random sample
            _values = np.random.uniform(
                0, 1, self.model.input.shape.as_list()[-1])
        # softmax policy
        if self.policy == 'softmax':
            # to pandas.Series
            _action = pd.Series(_values,
                                index=_returns.index,
                                name=_returns.name)
            return softmax(_action)
        # LONG best stock policy
        elif self.policy == 'best':
            # one-hot vector
            _action = np.zeros_like(_values).ravel()
            _action[np.argmax(_values)] = 1.0
            # to pandas.Series
            _action = pd.Series(_action,
                                index=_returns.index,
                                name=_returns.name)
            return _action

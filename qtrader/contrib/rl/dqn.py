import qtrader

import random
from collections import deque

import numpy as np
import tensorflow as tf


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # hyperparameters for DQN
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # replay memory
        self.memory = deque(maxlen=2500)

        # main and target models
        self.model = self.build_model()
        self.target_model = self.build_model()

        # init target model
        self.update_target_model()

    def build_model(self):
        X = tf.keras.layers.Input(shape=(self.state_size,))
        _fc = tf.keras.layers.Dense(
            50, activation='relu', kernel_initializer='he_uniform')(X)
        _fc = tf.keras.layers.Dense(
            50, activation='relu', kernel_initializer='he_uniform')(_fc)
        y = tf.keras.layers.Dense(
            self.action_size, activation='linear', kernel_initializer='he_uniform')(_fc)
        model = tf.keras.models.Model(X, y)
        model.compile(
            loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + \
                    self.gamma * np.amax(target_val[i])

        self.model.fit(update_input, target,
                       batch_size=batch_size, epochs=1, verbose=0)

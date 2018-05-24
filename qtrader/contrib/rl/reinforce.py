import qtrader

import random
from collections import deque

import numpy as np
import tensorflow as tf


class REINFORCEAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.lr = 0.001

        self.model = self.build_model()

        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):
        X = tf.keras.layers.Input(shape=(self.state_size,))
        _fc = tf.keras.layers.Dense(
            24, activation='relu', kernel_initializer='glorot_uniform')(X)
        _fc = tf.keras.layers.Dense(
            24, activation='relu', kernel_initializer='glorot_uniform')(_fc)
        y = tf.keras.layers.Dense(
            self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(_fc)
        model = tf.keras.models.Model(X, y)
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a).
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        model.summary()
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards = (
            discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

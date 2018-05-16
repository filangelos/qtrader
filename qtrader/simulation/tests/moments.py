from qtrader.utils.numpy import eps
from qtrader.simulation.tests.base import Test

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Moments(Test):

    @classmethod
    def run(cls, df_1, df_2, tolerance=0.1, render=False):
        # first order moment test
        first_order = cls._first_order_test(
            df_1, df_2, tolerance, render)
        # second order moment test
        second_order = cls._second_order_test(
            df_1, df_2, tolerance, render)
        # hypothesis test: accept if all tests passed
        hypothesis = first_order[0] and second_order[0]
        # summary of tests and scores
        info = {'first': first_order, 'second': second_order}
        return hypothesis, info

    @classmethod
    def _first_order_test(cls, df_1, df_2, tolerance, render):
        # column-wise mean values
        mu_1 = np.mean(df_1, axis=0)
        mu_2 = np.mean(df_2, axis=0)
        # relative deviation of mean values
        score = np.abs(mu_1 / (mu_2+eps) - 1)
        # plot distributions (kde)
        if render:
            # initialize figure & axes
            _, ax = plt.subplots()
            # iterate over columns
            for m in range(df_1.shape[1]):
                # distribution plot of family 1
                sns.distplot(df_1[:, m], label='df_1::%d' % m, ax=ax)
            # iterate over columns
            for m in range(df_2.shape[1]):
                # distribution plot of family 2
                sns.distplot(df_2[:, m], label='df_2::%d' % m, ax=ax)
            # plot settings
            ax.legend(ncol=2)
            ax.set_title('Distributions')
            ax.set_ylabel('Frequency')
        # threshold score
        return (score < tolerance).all(), score

    @classmethod
    def _second_order_test(cls, df_1, df_2, tolerance, render):
        # covariance matrix of family 1
        cov_1 = np.cov(df_1.T)
        # frobenius norm of family 1
        fro_1 = np.linalg.norm(cov_1, ord='fro')
        # covariance matrix of family 2
        cov_2 = np.cov(df_2.T)
        # frobenius norm of family 2
        fro_2 = np.linalg.norm(cov_2, ord='fro')
        # relative deviation of covariances
        # normalised by frobenius norms
        score = np.linalg.norm(cov_1 - cov_2, ord='fro') / \
            (np.sqrt(fro_1) * np.sqrt(fro_2) + eps)
        # plot covariance matrices
        if render:
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=3, figsize=(19.2, 4.8))
            # family 1
            sns.heatmap(cov_1, ax=axes[0])
            axes[0].set_title('Covariance Matrix: Series 1')
            # family 2
            sns.heatmap(cov_2, ax=axes[1])
            axes[1].set_title('Covariance Matrix: Series 2')
            # absolute difference
            sns.heatmap(np.abs(cov_1 - cov_2), ax=axes[2])
            axes[2].set_title('Covariance Matrix: Absolute Difference')
        # threshold score
        return score < tolerance, score

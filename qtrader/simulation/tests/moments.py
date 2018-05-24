from qtrader.utils.numpy import eps
from qtrader.simulation.tests.base import Test
from qtrader.utils import uuid

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Moments(Test):

    @classmethod
    def run(cls, df_1, df_2, tolerance=0.1, render=False):
        # generate UUID for render
        cls.UUID = uuid()
        # first order moment test
        first_order = cls._first_order_test(
            df_1, df_2, tolerance, render)
        # second order moment test
        second_order = cls._second_order_test(
            df_1, df_2, tolerance, render)
        # third order moment test
        third_order = cls._third_order_test(
            df_1, df_2, tolerance, render)
        # forth order moment test
        forth_order = cls._forth_order_test(
            df_1, df_2, tolerance, render)
        # hypothesis test: accept if all tests passed
        hypothesis = first_order[0] and second_order[0] and third_order[0] and forth_order[0]
        # summary of tests and scores
        info = {'first': first_order, 'second': second_order,
                'third': third_order, 'forth': forth_order}
        return hypothesis, info

    @classmethod
    def _first_order_test(cls, df_1, df_2, tolerance, render):
        # check dimensions consistency
        assert (df_1.columns == df_2.columns).all()
        # column-wise mean values
        mu_1 = np.mean(df_1, axis=0)
        mu_2 = np.mean(df_2, axis=0)
        # relative deviation of mean values
        score = np.abs(mu_1 / (mu_2+eps) - 1)
        # plot distributions
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=len(I), figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='df_1::%s' %
                             df_1.columns[m], color='g', norm_hist=True, ax=axes[i])
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='df_2::%s' %
                             df_1.columns[m], color='r', norm_hist=True, ax=axes[i])
            for i, ax in enumerate(axes):
                # plot settings
                ax.legend()
                ax.set_title('%s: Distributions' % df_1.columns[I[i]])
                ax.set_ylabel('Frequency')
            fig.savefig('assets/tmp/first_order_%s.pdf' % cls.UUID)
            # fig.show()
        # threshold score
        return (score < tolerance).all(), score

    @classmethod
    def _second_order_test(cls, df_1, df_2, tolerance, render):
        # check dimensions consistency
        assert (df_1.columns == df_2.columns).all()
        # covariance matrix of family 1
        cov_1 = np.cov(df_1.T)
        # frobenius norm of family 1
        fro_1 = np.linalg.norm(cov_1, ord='fro')
        # covariance matrix of family 2
        cov_2 = np.cov(df_2.T)
        # frobenius norm of family 2
        fro_2 = np.linalg.norm(cov_2, ord='fro')
        # relative deviation of covariances
        # normalized by frobenius norms
        score = np.linalg.norm(cov_1 - cov_2, ord='fro') / \
            (np.sqrt(fro_1) * np.sqrt(fro_2) + eps)
        # plot covariance matrices
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # fetch sub-matrices
            sub_cov_1 = cov_1[np.ix_(I, I)]
            sub_cov_2 = cov_2[np.ix_(I, I)]
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=3, figsize=(19.2, 4.8))
            # family 1
            sns.heatmap(sub_cov_1, xticklabels=I, yticklabels=I, ax=axes[0])
            axes[0].set_title('Covariance Matrix: Series 1')
            # family 2
            sns.heatmap(sub_cov_2, xticklabels=I, yticklabels=I, ax=axes[1])
            axes[1].set_title('Covariance Matrix: Series 2')
            # absolute difference
            sns.heatmap(np.abs(sub_cov_1 - sub_cov_2),
                        xticklabels=I, yticklabels=I, ax=axes[2])
            axes[2].set_title('Covariance Matrix: Absolute Difference')
            fig.savefig('assets/tmp/second_order_%s.pdf' % cls.UUID)
            # fig.show()
        # threshold score
        return score < tolerance, score

    @classmethod
    def _third_order_test(cls, df_1, df_2, tolerance, render):
        # check dimensions consistency
        assert (df_1.columns == df_2.columns).all()
        # skewness of family 1
        skew_1 = df_1.skew()
        # skewness of family 2
        skew_2 = df_2.skew()
        # relative deviation of kurtosis
        score = np.abs(skew_1 / (skew_2+eps) - 1)
        # plot mean-median deviations
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=len(I), figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='df_1::%s' %
                             df_1.columns[m], color='g', hist=False, norm_hist=True, ax=axes[i])
                # vertical line for mean of family 1
                axes[i].vlines(df_1.iloc[:, m].mean(), 0, 1e20, label='df_1::mean::%s' %
                               df_1.columns[m], color='g', linestyles='-')
                # vertical line for median of family 1
                axes[i].vlines(df_1.iloc[:, m].median(), 0, 1e20, label='df_1::median::%s' %
                               df_1.columns[m], color='g', linestyles='-.')
            for i, m in enumerate(I):
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='df_2::%s' %
                             df_1.columns[m], color='r', hist=False, norm_hist=True, ax=axes[i])
                # vertical line for mean of family 2
                axes[i].vlines(df_2.iloc[:, m].mean(), 0, 1e20, label='df_2::mean::%s' %
                               df_2.columns[m], color='r', linestyles='-')
                # vertical line for median of family 2
                axes[i].vlines(df_2.iloc[:, m].median(), 0, 1e20, label='df_2::median::%s' %
                               df_2.columns[m], color='r', linestyles='-.')
            for i, ax in enumerate(axes):
                # plot settings
                ax.legend()
                ax.set_title('%s: Distributions' % df_1.columns[I[i]])
                ax.set_ylabel('Frequency')
            fig.savefig('assets/tmp/third_order_%s.pdf' % cls.UUID)
            # fig.show()
        # threshold score
        return (score < tolerance).all(), score

    @classmethod
    def _forth_order_test(cls, df_1, df_2, tolerance, render):
        # check dimensions consistency
        assert (df_1.columns == df_2.columns).all()
        # kurtosis of family 1
        kurt_1 = df_1.kurt()
        # kurtosis of family 2
        kurt_2 = df_2.kurt()
        # relative deviation of kurtosis
        score = np.abs(kurt_1 / (kurt_2+eps) - 1)
        # plot y-axis log scale
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=len(I), figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='df_1::%s' %
                             df_1.columns[m], color='g', hist_kws={'log': True}, norm_hist=True, ax=axes[i])
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='df_2::%s' %
                             df_2.columns[m], color='r', hist_kws={'log': True}, norm_hist=True, ax=axes[i])
            for i, ax in enumerate(axes):
                # plot settings
                ax.legend()
                ax.set_title('%s: Distributions' % df_1.columns[I[i]])
                ax.set_ylabel('Frequency')
            fig.savefig('assets/tmp/forth_order_%s.pdf' % cls.UUID)
            # fig.show()
        # threshold score
        return (score < tolerance).all(), score

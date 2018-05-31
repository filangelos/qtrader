from qtrader.utils.numpy import eps
from qtrader.simulation.tests.base import Test
from qtrader.utils import uuid

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns


class Moments(Test):

    @classmethod
    def run(cls, df_1, df_2, log=False, render=False):
        # generate UUID for render
        cls.UUID = uuid()
        # first order moment test
        first_order = cls._first_order_test(
            df_1, df_2, log, render)
        # second order moment test
        second_order = cls._second_order_test(
            df_1, df_2, log, render)
        # third order moment test
        third_order = cls._third_order_test(
            df_1, df_2, log, render)
        # forth order moment test
        forth_order = cls._forth_order_test(
            df_1, df_2, log, render)
        # summary of tests and scores
        info = {'first': first_order, 'second': second_order,
                'third': third_order, 'forth': forth_order}
        return info

    @classmethod
    def _first_order_test(cls, df_1, df_2, tolerance, render):
        """First order moments comparison test.

        Parameters
        ----------
        df_1: pandas.DataFrame
            Family 1 data.
        df_2: pandas.DataFrame
            Family 2 data.
        log: bool, optional
            Flag for printing summary.
        render: bool, optional
            Flag for printing distributions.

        Returns
        -------
        summary: dict
            First order test summary.
        """
        # type check
        assert isinstance(df_1, pd.DataFrame)
        assert isinstance(df_2, pd.DataFrame)
        # interface check
        assert (df_1.columns == df_2.columns).all()
        # column-wise mean values
        mu_1 = df_1.mean(axis=0)
        mu_2 = df_2.mean(axis=0)
        # relative deviation of mean values
        dmu = np.abs((mu_1+eps) / (mu_2+eps) - 1)
        # column-wise median values
        median_1 = df_1.median(axis=0)
        median_2 = df_2.median(axis=0)
        dmedian = np.abs((median_1+eps) / (median_2+eps) - 1)
        # column-size mode values
        mode_1 = df_1.mode(axis=0).iloc[0]
        mode_2 = df_2.mode(axis=0).iloc[0]
        dmode = np.abs((mode_1+eps) / (mode_2+eps) - 1)
        # summary table for each ticker
        summary = {}
        # iterate over tickers
        for ticker in df_1:
            # summary table columns
            _name1 = 'series 1'
            _name2 = 'series 2'
            _name3 = 'deviations'
            summary[ticker] = pd.DataFrame({_name1: [mu_1[ticker], median_1[ticker], mode_1[ticker]],
                                            _name2: [mu_2[ticker], median_2[ticker], mode_1[ticker]],
                                            _name3: [dmu[ticker], dmedian[ticker], dmode[ticker]]},
                                           columns=[_name1, _name2, _name3],
                                           index=['mean', 'median', 'mode'])
        # print summary
        if log:
            for ticker, summ in summary.items():
                print(ticker)
                print(summ)
                print('\n')
        # plot distributions
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(
                ncols=len(I), nrows=2, sharex=True, figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='%s' %
                             df_1.columns[m], color='g', norm_hist=True, ax=axes[0, i])
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='%s' %
                             df_1.columns[m], color='r', norm_hist=True, ax=axes[1, i])
                # y-axis limits
                ymin_1, ymax_1 = axes[0, i].get_ylim()
                ymin_2, ymax_2 = axes[1, i].get_ylim()
                ymin = min(ymin_1, ymin_2)
                ymax = max(ymax_1, ymax_2)
                # vertical line for mean of family 1
                axes[0, i].vlines(mu_1.iloc[m], ymin, ymax, label='%s::mean' %
                                  df_1.columns[m], color='g', linestyles='-')
                # vertical line for median of family 1
                axes[0, i].vlines(median_1.iloc[m], ymin, ymax, label='%s::median' %
                                  df_1.columns[m], color='g', linestyles='-.')
                # vertical line for mode of family 1
                axes[0, i].vlines(mode_1.iloc[m], ymin, ymax, label='%s::mode' %
                                  df_1.columns[m], color='g', linestyles='--')
                # vertical line for mean of family 2
                axes[1, i].vlines(mu_2.iloc[m], ymin, ymax, label='%s::mean' %
                                  df_2.columns[m], color='r', linestyles='-')
                # vertical line for median of family 2
                axes[1, i].vlines(median_2.iloc[m], ymin, ymax, label='%s::median' %
                                  df_2.columns[m], color='r', linestyles='-.')
                # vertical line for mode of family 2
                axes[1, i].vlines(mode_2.iloc[m], ymin, ymax, label='%s::mode' %
                                  df_2.columns[m], color='r', linestyles='--')
                # setting for family 1
                axes[0, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[0, i].legend()
                # settings for family 2
                axes[1, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[1, i].legend()
            # present figure
            fig.show()
        return summary

    @classmethod
    def _second_order_test(cls, df_1, df_2, tolerance, render):
        """Second order moments comparison test.

        Parameters
        ----------
        df_1: pandas.DataFrame
            Family 1 data.
        df_2: pandas.DataFrame
            Family 2 data.
        log: bool, optional
            Flag for printing summary.
        render: bool, optional
            Flag for printing distributions.

        Returns
        -------
        summary: dict
            Second order test summary.
        """
        # type check
        assert isinstance(df_1, pd.DataFrame)
        assert isinstance(df_2, pd.DataFrame)
        # interface check
        assert (df_1.columns == df_2.columns).all()
        # covariance matrix of family 1
        cov_1 = df_1.cov()
        # frobenius norm of family 1
        fro_1 = np.linalg.norm(cov_1, ord='fro')
        # covariance matrix of family 2
        cov_2 = df_2.cov()
        # frobenius norm of family 2
        fro_2 = np.linalg.norm(cov_2, ord='fro')
        # relative deviation of covariances
        # normalized by frobenius norms
        dcov = np.linalg.norm(cov_1 - cov_2, ord='fro') / \
            (np.sqrt(fro_1) * np.sqrt(fro_2) + eps)
        # column-wise std values
        std_1 = df_1.std(axis=0)
        std_2 = df_2.std(axis=0)
        # relative deviation of std values
        dstd = np.abs((std_1+eps) / (std_2+eps) - 1)
        # summary table for each ticker
        summary = {}
        # iterate over tickers
        for ticker in df_1:
            # Bartlett test
            bartlett_s, bartlett_p = scipy.stats.bartlett(
                df_1[ticker], df_2[ticker])
            # Levene test
            levene_s, levene_p = scipy.stats.levene(df_1[ticker], df_2[ticker])
            # Fligner-Killeen test
            fligner_s, fligner_p = scipy.stats.fligner(
                df_1[ticker], df_2[ticker])
            # summary table columns
            _name1 = 'series 1'
            _name2 = 'series 2'
            _name3 = 'deviations'
            _name4 = 'statistic'
            _name5 = 'p-value (>0.05)'
            summary[ticker] = pd.DataFrame({_name1: [std_1[ticker], None, None, None],
                                            _name2: [std_2[ticker], None, None, None],
                                            _name3: [dstd[ticker], None, None, None],
                                            _name4: [None, bartlett_s, levene_s, fligner_s],
                                            _name5: [None, bartlett_p, levene_p, fligner_p]},
                                           columns=[_name1, _name2,
                                                    _name3, _name4, _name5],
                                           index=['std', 'Bartlett', 'Levene', 'Fligner-Killeen'])
        # print summary
        if log:
            for ticker, summ in summary.items():
                print(ticker)
                print(summ)
                print('\n')
        # plot covariance matrices
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # fetch sub-matrices
            sub_cov_1 = cov_1.values[np.ix_(I, I)]
            sub_cov_2 = cov_2.values[np.ix_(I, I)]
            # half-cov mask
            mask = np.ones_like(sub_cov_1)
            mask[np.triu_indices_from(mask)] = False
            # initialize figure & axes
            fig, axes = plt.subplots(ncols=3, figsize=(19.2, 4.8))
            # family 1
            sns.heatmap(sub_cov_1, xticklabels=df_1.columns[I], yticklabels=df_1.columns[I],
                        ax=axes[0], mask=mask, cmap=plt.cm.Greys)
            axes[0].set(title='Covariance Matrix: Series 1')
            # family 2
            sns.heatmap(sub_cov_2, xticklabels=df_2.columns[I], yticklabels=df_2.columns[I],
                        ax=axes[1], mask=mask, cmap=plt.cm.Greys)
            axes[1].set(title='Covariance Matrix: Series 2')
            # absolute difference
            sns.heatmap(np.abs(sub_cov_1 - sub_cov_2),
                        xticklabels=df_1.columns[I], yticklabels=df_1.columns[I],
                        ax=axes[2], mask=mask, cmap=plt.cm.Greys)
            axes[2].set(title='Covariance Matrix: Absolute Difference')
            # present figure
            fig.show()
        return summary

    @classmethod
    def _third_order_test(cls, df_1, df_2, tolerance, render):
        """Third order moments comparison test.

        Parameters
        ----------
        df_1: pandas.DataFrame
            Family 1 data.
        df_2: pandas.DataFrame
            Family 2 data.
        log: bool, optional
            Flag for printing summary.
        render: bool, optional
            Flag for printing distributions.

        Returns
        -------
        summary: dict
            Third order test summary.
        """
        # type check
        assert isinstance(df_1, pd.DataFrame)
        assert isinstance(df_2, pd.DataFrame)
        # interface check
        assert (df_1.columns == df_2.columns).all()
        # column-wise skewness values
        skew_1 = df_1.skew(axis=0)
        skew_2 = df_2.skew(axis=0)
        # relative deviation of skewness values
        dskew = np.abs((skew_1+eps) / (skew_2+eps) - 1)
        # summary table for each ticker
        summary = {}
        # iterate over tickers
        for ticker in df_1:
            # Skewness test for family 1
            skewtest_s_1, skewtest_p_1 = scipy.stats.skewtest(df_1[ticker])
            # Skewness test for family 2
            skewtest_s_2, skewtest_p_2 = scipy.stats.skewtest(df_2[ticker])
            # summary table columns
            _name1 = 'series 1'
            _name2 = 'series 2'
            _name3 = 'deviations'
            summary[ticker] = pd.DataFrame({_name1: [skew_1[ticker], skewtest_s_1, skewtest_p_1],
                                            _name2: [skew_2[ticker], skewtest_s_2, skewtest_p_2],
                                            _name3: [dskew[ticker], None, None]},
                                           columns=[_name1, _name2, _name3],
                                           index=['skewness', 'skewtest statistic', 'skewtest p-value (>0.05)'])
        # print summary
        if log:
            for ticker, summ in summary.items():
                print(ticker)
                print(summ)
                print('\n')
        # plot distributions
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(
                ncols=len(I), nrows=2, sharex=True, figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # normal distribution with family 1's statistics
                mu_1 = df_1.iloc[:, m].mean()
                std_1 = df_1.iloc[:, m].std()
                norm_1 = np.random.normal(
                    mu_1, std_1, 10*df_1.iloc[:, m].count())
                sns.distplot(norm_1, label='$\mathcal{N}(%.4f, %.4f)$' %
                             (mu_1, std_1), color='k', norm_hist=True, ax=axes[0, i])
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='%s' %
                             df_1.columns[m], color='g', norm_hist=True, ax=axes[0, i])
                # normal distribution with family 2's statistics
                mu_2 = df_2.iloc[:, m].mean()
                std_2 = df_2.iloc[:, m].std()
                norm_2 = np.random.normal(
                    mu_2, std_2, 10*df_2.iloc[:, m].count())
                sns.distplot(norm_2, label='$\mathcal{N}(%.4f, %.4f)$' %
                             (mu_2, std_2), color='k', norm_hist=True, ax=axes[1, i])
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='%s' %
                             df_1.columns[m], color='r', norm_hist=True, ax=axes[1, i])
                # y-axis limits
                ymin_1, ymax_1 = axes[0, i].get_ylim()
                ymin_2, ymax_2 = axes[1, i].get_ylim()
                ymin = min(ymin_1, ymin_2)
                ymax = max(ymax_1, ymax_2)
                # setting for family 1
                axes[0, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[0, i].legend()
                # settings for family 2
                axes[1, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[1, i].legend()
            # present figure
            fig.show()
        return summary

    @classmethod
    def _forth_order_test(cls, df_1, df_2, tolerance, render):
        """Forth order moments comparison test.

        Parameters
        ----------
        df_1: pandas.DataFrame
            Family 1 data.
        df_2: pandas.DataFrame
            Family 2 data.
        log: bool, optional
            Flag for printing summary.
        render: bool, optional
            Flag for printing distributions.

        Returns
        -------
        summary: dict
            Forth order test summary.
        """
        # type check
        assert isinstance(df_1, pd.DataFrame)
        assert isinstance(df_2, pd.DataFrame)
        # interface check
        assert (df_1.columns == df_2.columns).all()
        # column-wise kurtosis values
        kurt_1 = df_1.kurt(axis=0)
        kurt_2 = df_2.kurt(axis=0)
        # relative deviation of kurtosis values
        dkurt = np.abs((kurt_1+eps) / (kurt_2+eps) - 1)
        # summary table for each ticker
        summary = {}
        # iterate over tickers
        for ticker in df_1:
            # Kurtosis test for family 1
            kurtosistest_s_1, kurtosistest_p_1 = scipy.stats.kurtosistest(
                df_1[ticker])
            # Kurtosis test for family 2
            kurtosistest_s_2, kurtosistest_p_2 = scipy.stats.kurtosistest(
                df_2[ticker])
            # summary table columns
            _name1 = 'series 1'
            _name2 = 'series 2'
            _name3 = 'deviations'
            summary[ticker] = pd.DataFrame({_name1: [kurt_1[ticker], kurtosistest_s_1, kurtosistest_p_1],
                                            _name2: [kurt_2[ticker], kurtosistest_s_2, kurtosistest_p_2],
                                            _name3: [dkurt[ticker], None, None]},
                                           columns=[_name1, _name2, _name3],
                                           index=['skewness', 'kurtosistest statistic', 'kurtosistest p-value (>0.05)'])
        # print summary
        if log:
            for ticker, summ in summary.items():
                print(ticker)
                print(summ)
                print('\n')
        # plot distributions
        if render:
            # random selection of assets to render
            I = np.sort(np.random.choice(
                df_1.shape[1], min(3, df_1.shape[1]), replace=False))
            # initialize figure & axes
            fig, axes = plt.subplots(
                ncols=len(I), nrows=2, sharex=True, figsize=(6.4 * len(I), 4.8))
            for i, m in enumerate(I):
                # normal distribution with family 1's statistics
                mu_1 = df_1.iloc[:, m].mean()
                std_1 = df_1.iloc[:, m].std()
                norm_1 = np.random.normal(
                    mu_1, std_1, 10*df_1.iloc[:, m].count())
                sns.distplot(norm_1, label=r'$\mathcal{N}(%.4f, %.4f)$' %
                             (mu_1, std_1), color='k', norm_hist=True, ax=axes[0, i])
                # distribution plot of family 1
                sns.distplot(df_1.iloc[:, m], label='%s' %
                             df_1.columns[m], color='g', norm_hist=True, ax=axes[0, i])
                # normal distribution with family 2's statistics
                mu_2 = df_2.iloc[:, m].mean()
                std_2 = df_2.iloc[:, m].std()
                norm_2 = np.random.normal(
                    mu_2, std_2, 10*df_2.iloc[:, m].count())
                sns.distplot(norm_2, label=r'$\mathcal{N}(%.4f, %.4f)$' %
                             (mu_2, std_2), color='k', norm_hist=True, ax=axes[1, i])
                # distribution plot of family 2
                sns.distplot(df_2.iloc[:, m], label='%s' %
                             df_1.columns[m], color='r', norm_hist=True, ax=axes[1, i])
                # y-axis limits
                ymin_1, ymax_1 = axes[0, i].get_ylim()
                ymin_2, ymax_2 = axes[1, i].get_ylim()
                ymin = min(ymin_1, ymin_2)
                ymax = max(ymax_1, ymax_2)
                # setting for family 1
                axes[0, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[0, i].legend()
                # settings for family 2
                axes[1, i].set(ylabel='Frequency',
                               xlabel='', ylim=[ymin, ymax])
                axes[1, i].legend()
            # present figure
            fig.show()
        return summary

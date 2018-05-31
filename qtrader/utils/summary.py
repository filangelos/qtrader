import pandas as pd
import matplotlib.pyplot as plt

import qtrader


def stats(returns):
    """Generate statistics report for strategy.

    Parameters
    ----------
    prices: pandas.DataFrame
        Prices of asset universe.
    returns: pandas.Series
        Realised returns of strategy.
    weights: pandas.DataFrame
        Portfolio weights of strategy.

    Returns
    -------
    table: pd.Series
        Strategy report.
    """
    report = {
        'Mean Returns': qtrader.utils.econometric.mean_returns(returns),
        'Cumulative Returns': qtrader.utils.econometric.cum_returns(returns).iloc[-1],
        'Volatility': qtrader.utils.econometric.std_returns(returns),
        'Sharpe Ratio': qtrader.utils.econometric.sharpe_ratio(returns),
        'Max Drawdown': qtrader.utils.econometric.max_drawdown(returns).iloc[-1],
        'Average Drawdown Time': qtrader.utils.econometric.average_drawdown_time(returns).days,
        'Skewness': qtrader.utils.econometric.skewness(returns),
        'Kurtosis': qtrader.utils.econometric.kurtosis(returns),
        'Tail Ratio': qtrader.utils.econometric.tail_ratio(returns),
        'Value at Risk': qtrader.utils.econometric.value_at_risk(returns),
        'Conditional Value at Risk': qtrader.utils.econometric.conditional_value_at_risk(returns),
        'Hit Ratio': qtrader.utils.econometric.hit_ratio(returns),
        'Average Win to Average Loss': qtrader.utils.econometric.awal(returns),
        'Average Profitability Per Trade': qtrader.utils.econometric.appt(returns)
    }
    table = pd.Series(
        report,
        name=(returns.name or 'Strategy'),
        dtype=object
    )
    return table


def figure(prices, returns, weights, path=None):
    """Generate statistics figures for strategy.

    Parameters
    ----------
    prices: pandas.DataFrame
        Prices of asset universe.
    returns: pandas.Series
        Realised returns of strategy.
    weights: pandas.DataFrame
        Portfolio weights of strategy.
    path: str, optional
        Path to store figure.
    """
    qtrader.utils.plotting.drawdown(returns, path)
    for ticker in prices:
        qtrader.utils.plotting.trades(prices[ticker], weights[ticker], path)

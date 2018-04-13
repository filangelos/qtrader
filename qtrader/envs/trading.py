from qtrader.envs.base import BaseEnv
from qtrader.envs.data_loader import Finance

import pandas as pd


class TradingEnv(BaseEnv):
    """OpenAI Gym Trading Environment with Daily Returns Reward."""

    def _get_data(self, **kwargs):
        return Finance.Returns(self.universe, freq=self.trading_period, **kwargs)

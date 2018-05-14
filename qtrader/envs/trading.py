from qtrader.envs.base import BaseEnv
from qtrader.envs.data_loader import Finance

import pandas as pd


class TradingEnv(BaseEnv):
    """OpenAI Gym Trading Environment with Daily Returns Reward."""

    def _get_prices(self, **kwargs) -> pd.DataFrame:
        return Finance.Prices(self.universe, freq=self.trading_period, **kwargs)

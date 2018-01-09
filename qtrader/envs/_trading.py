from __future__ import absolute_import

from qtrader.envs._base import TradingEnv
from qtrader.adapter import Finance
import qtrader


class DailyReturnEnv(TradingEnv):
    """OpenAI Gym Trading Environment with Daily Returns Reward."""

    def _get_data(self, **kwargs):
        return Finance.Returns(self.universe, **kwargs)

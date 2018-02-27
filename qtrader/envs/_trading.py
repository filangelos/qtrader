from qtrader.envs._base import TradingEnv
from qtrader.adapters import Finance


class DailyReturnEnv(TradingEnv):
    """OpenAI Gym Trading Environment with Daily Returns Reward."""

    def _get_data(self, **kwargs):
        return Finance.Returns(self.universe, **kwargs)

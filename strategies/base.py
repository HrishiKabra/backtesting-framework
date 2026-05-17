from abc import ABC, abstractmethod
import pandas as pd
from core.engine import LookaheadBarrier


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        """
        Generate target weight signals for all dates in the data.

        Called ONCE before the BacktestEngine execution loop.
        Returns a DataFrame: index=trading_dates, columns=tickers,
        values=target_weights in [-1, 1].

        Must not contain NaN values. Training/warmup periods should be 0.0.
        Weights represent fraction of portfolio NAV: 0.10 = 10% long,
        -0.05 = 5% short.
        """

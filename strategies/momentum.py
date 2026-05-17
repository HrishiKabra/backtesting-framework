import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum: monthly rebalance, long top quintile,
    short bottom quintile ranked by lookback-minus-skip month return.

    Academic basis: Jegadeesh & Titman (1993).
    """

    param_grid = {
        "lookback_months": [6, 9, 12, 15],
        "skip_months": [1, 2],
    }

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        self.lookback_days = _p.get("lookback_months", 12) * 21
        self.skip_days = _p.get("skip_months", 1) * 21

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        ret_lookback = close.pct_change(self.lookback_days)
        ret_skip = close.pct_change(self.skip_days)
        momentum = ret_lookback - ret_skip

        month_ends = close.resample("BME").last().index

        for date in month_ends:
            if date not in close.index:
                continue
            scores = momentum.loc[date].dropna()
            if len(scores) < 5:
                continue

            n = len(scores)
            quintile_size = max(1, n // 5)

            ranked = scores.rank(ascending=True)
            long_tickers = ranked.nlargest(quintile_size).index
            short_tickers = ranked.nsmallest(quintile_size).index

            long_weight = 0.5 / len(long_tickers)
            short_weight = -0.5 / len(short_tickers)

            signals.loc[date, :] = 0.0
            for ticker in long_tickers:
                if ticker in signals.columns:
                    signals.loc[date, ticker] = long_weight
            for ticker in short_tickers:
                if ticker in signals.columns:
                    signals.loc[date, ticker] = short_weight

        signals = signals.ffill().fillna(0.0)
        signals.iloc[:self.lookback_days] = 0.0

        return signals

import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum: monthly rebalance, long top quintile,
    short bottom quintile ranked by 12-minus-1 month return.

    Academic basis: Jegadeesh & Titman (1993) — stocks with strong 12-month
    returns continue to outperform over the next 3-12 months. Skipping the
    most recent month avoids short-term mean-reversion.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # 12-month return (252 days) and 1-month return (21 days)
        ret_12m = close.pct_change(252)
        ret_1m = close.pct_change(21)
        momentum = ret_12m - ret_1m  # 12-minus-1 month return

        # Rebalance on last business day of each month
        month_ends = close.resample("BME").last().index

        for date in month_ends:
            if date not in close.index:
                continue
            scores = momentum.loc[date].dropna()
            if len(scores) < 5:
                continue  # need enough stocks to rank

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

        # Forward-fill signals between rebalance dates
        signals = signals.replace(0.0, np.nan)
        signals = signals.ffill().fillna(0.0)

        # Zero out warmup period (need 252 days for 12-month return)
        signals.iloc[:252] = 0.0

        return signals

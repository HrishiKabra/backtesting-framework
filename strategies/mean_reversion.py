import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class BollingerStrategy(Strategy):
    """
    Mean-reversion using Bollinger Bands (20-day rolling mean ± 2σ).
    Long when price < lower band, short when price > upper band.
    Exit when price crosses back through the mean.
    Stop-loss at rolling mean ±3σ (trailing stop).

    Academic basis: Bollinger (2001) — prices oscillate around a moving
    average; extremes signal reversion opportunities.
    """

    def __init__(self, config: BacktestConfig, window: int = 20, n_std: float = 2.0):
        self.config = config
        self.window = window
        self.n_std = n_std

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        upper = rolling_mean + self.n_std * rolling_std
        lower = rolling_mean - self.n_std * rolling_std
        stop_upper = rolling_mean + 3.0 * rolling_std
        stop_lower = rolling_mean - 3.0 * rolling_std

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        # Track current signal direction per ticker
        current_signal = {t: 0.0 for t in close.columns}

        for i, date in enumerate(close.index):
            if i < self.window:
                continue
            for ticker in close.columns:
                price = close.loc[date, ticker]
                if pd.isna(price):
                    continue

                mean = rolling_mean.loc[date, ticker]
                lb = lower.loc[date, ticker]
                ub = upper.loc[date, ticker]
                sl_lo = stop_lower.loc[date, ticker]
                sl_hi = stop_upper.loc[date, ticker]

                if pd.isna(mean):
                    continue

                sig = current_signal[ticker]

                # Stop-loss: exit if price hits 3σ stop
                if sig > 0 and price < sl_lo:
                    sig = 0.0
                elif sig < 0 and price > sl_hi:
                    sig = 0.0

                # Exit: price crosses through mean
                if sig > 0 and price >= mean:
                    sig = 0.0
                elif sig < 0 and price <= mean:
                    sig = 0.0

                # Entry
                if sig == 0.0:
                    if price < lb:
                        sig = 0.1
                    elif price > ub:
                        sig = -0.1

                current_signal[ticker] = sig
                signals.loc[date, ticker] = sig

        return signals.fillna(0.0)

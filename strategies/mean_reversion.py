import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class BollingerStrategy(Strategy):
    """
    Mean-reversion using Bollinger Bands.
    Long when price < lower band, short when price > upper band.
    Exit when price crosses back through the mean.
    Stop-loss at rolling mean ±3σ (trailing stop).

    Academic basis: Bollinger (2001).
    """

    param_grid = {
        "window": [10, 15, 20, 25, 30],
        "entry_z": [1.5, 2.0, 2.5, 3.0],
    }

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        self.window = _p.get("window", 20)
        self.entry_z = _p.get("entry_z", 2.0)

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        upper = rolling_mean + self.entry_z * rolling_std
        lower = rolling_mean - self.entry_z * rolling_std
        stop_upper = rolling_mean + 3.0 * rolling_std
        stop_lower = rolling_mean - 3.0 * rolling_std

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
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

                stopped = False
                if sig > 0 and price < sl_lo:
                    sig = 0.0
                    stopped = True
                elif sig < 0 and price > sl_hi:
                    sig = 0.0
                    stopped = True

                if sig > 0 and price >= mean:
                    sig = 0.0
                elif sig < 0 and price <= mean:
                    sig = 0.0

                if not stopped and sig == 0.0:
                    if price < lb:
                        sig = 0.1
                    elif price > ub:
                        sig = -0.1

                current_signal[ticker] = sig
                signals.loc[date, ticker] = sig

        return signals.fillna(0.0)

import pandas as pd
import numpy as np
from typing import List, Tuple
from statsmodels.tsa.stattools import coint
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


def _kalman_hedge_ratio(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Estimates dynamic hedge ratio β via scalar Kalman filter.
    State: β_t, Observation: y_t = β_t * x_t + ε_t
    """
    n = len(y)
    beta = np.zeros(n)
    P = np.ones(n)
    Q = 1e-5   # process noise (how fast β changes)
    R = 1.0    # observation noise

    beta[0] = y[0] / x[0] if abs(x[0]) > 1e-8 else 1.0
    for t in range(1, n):
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Q
        H = x[t]
        S = H * P_pred * H + R
        K = P_pred * H / S if abs(S) > 1e-10 else 0.0
        beta[t] = beta_pred + K * (y[t] - H * beta_pred)
        P[t] = (1 - K * H) * P_pred

    return beta


class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage via cointegrated pairs.
    Uses a scalar Kalman filter to estimate the dynamic hedge ratio β.
    Spread = price_A - β * price_B, traded when z-score exceeds ±2σ.

    Academic basis: Engle & Granger (1987) — cointegrated pairs share a
    long-run equilibrium; deviations from it mean-revert.
    """

    DEFAULT_PAIRS = [
        ("AAPL", "MSFT"), ("JPM", "GS"), ("XOM", "CVX"),
        ("NVDA", "AMD"), ("WMT", "HD"),
    ]

    param_grid = {"entry_z": [1.5, 2.0, 2.5, 3.0], "exit_z": [0.0, 0.5]}

    def __init__(
        self,
        config: BacktestConfig,
        params: dict = None,
        pairs: List[Tuple[str, str]] = None,
        training_days: int = 504,  # ~2 years
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        z_stop: float = 3.0,
        spread_window: int = 20,
    ):
        _p = params or {}
        self.config = config
        self.pairs = pairs if pairs is not None else self.DEFAULT_PAIRS
        self.training_days = training_days
        self.z_entry = _p.get("entry_z", z_entry)
        self.z_exit = _p.get("exit_z", z_exit)
        self.z_stop = z_stop
        self.spread_window = spread_window

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()  # use .ffill() not fillna(method="ffill") — modern pandas

        all_tickers = list(close.columns)
        signals = pd.DataFrame(0.0, index=close.index, columns=all_tickers)

        for ticker_a, ticker_b in self.pairs:
            if ticker_a not in close.columns or ticker_b not in close.columns:
                continue

            y = close[ticker_a].values
            x = close[ticker_b].values

            # Cointegration test on training period only
            train_y = y[:self.training_days]
            train_x = x[:self.training_days]
            valid = ~(np.isnan(train_y) | np.isnan(train_x))
            if valid.sum() < 60:
                continue

            _, pvalue, _ = coint(train_y[valid], train_x[valid])
            if pvalue >= 0.05:
                continue  # not cointegrated — skip pair

            # Kalman filter on full history for dynamic β
            valid_full = ~(np.isnan(y) | np.isnan(x))
            beta_full = np.ones(len(y))
            beta_full[valid_full] = _kalman_hedge_ratio(
                y[valid_full], x[valid_full]
            )
            # Forward fill beta to all dates
            beta_series = pd.Series(beta_full, index=close.index)
            beta_series[~valid_full] = np.nan
            beta_series = beta_series.ffill().fillna(1.0)

            spread = pd.Series(y, index=close.index) - beta_series * pd.Series(x, index=close.index)
            roll_mean = spread.rolling(self.spread_window).mean()
            roll_std = spread.rolling(self.spread_window).std()
            z_score = (spread - roll_mean) / roll_std.replace(0, np.nan)
            z_score = z_score.fillna(0.0)

            # Generate signals for this pair
            pair_signal = pd.Series(0.0, index=close.index)
            current_sig = 0.0
            w = 0.1  # weight per leg

            for i, date in enumerate(close.index):
                if i < self.training_days:
                    pair_signal[date] = 0.0
                    continue

                z = z_score[date]
                if pd.isna(z):
                    pair_signal[date] = current_sig
                    continue

                # Stop loss
                stopped = False
                if current_sig > 0 and z <= -self.z_stop:
                    current_sig = 0.0
                    stopped = True
                elif current_sig < 0 and z >= self.z_stop:
                    current_sig = 0.0
                    stopped = True

                # Exit
                if current_sig > 0 and z >= self.z_exit:
                    current_sig = 0.0
                elif current_sig < 0 and z <= self.z_exit:
                    current_sig = 0.0

                # Entry — only if not stopped on this bar
                if not stopped and current_sig == 0.0:
                    if z < -self.z_entry:
                        current_sig = 1.0   # long spread
                    elif z > self.z_entry:
                        current_sig = -1.0  # short spread

                pair_signal[date] = current_sig

            # Bake hedge ratio into weights:
            # long spread: long A (+w), short B (-w*β)
            # short spread: short A (-w), long B (+w*β)
            b = beta_series
            signals[ticker_a] += pair_signal * w
            signals[ticker_b] += pair_signal * (-w * b)

        return signals.fillna(0.0)

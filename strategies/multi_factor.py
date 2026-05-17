import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row cross-sectionally (subtract mean, divide by std)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


class MultiFactorStrategy(Strategy):
    """
    Monthly-rebalanced long-short factor model combining three factors,
    all cross-sectionally z-scored and equally blended:
      - Momentum (12-1 month return), weight 0.5
      - Low Volatility (negative 20-day realized vol), weight 0.3
      - Short-term Reversal (negative 1-month return), weight 0.2

    Long top quintile, short bottom quintile.

    Note: P/B and ROE omitted — yfinance fundamental data is unreliable
    for backtesting. Low-vol (Ang et al. 2006) and reversal (Jegadeesh 1990)
    are academically validated replacements.
    """

    FACTOR_WEIGHTS = {"momentum": 0.5, "low_vol": 0.3, "reversal": 0.2}
    param_grid = {"momentum_wt": [0.33, 0.5, 0.7], "lowvol_wt": [0.1, 0.3, 0.5]}

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        momentum_wt = _p.get("momentum_wt", 0.5)
        lowvol_wt = _p.get("lowvol_wt", 0.3)
        reversal_wt = round(1.0 - momentum_wt - lowvol_wt, 10)
        self.factor_weights = {"momentum": momentum_wt, "low_vol": lowvol_wt, "reversal": reversal_wt}

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()  # use .ffill() not fillna(method="ffill") — modern pandas

        ret_12m = close.pct_change(252)
        ret_1m = close.pct_change(21)
        momentum_raw = ret_12m - ret_1m
        low_vol_raw = -close.pct_change().rolling(20).std()
        reversal_raw = -ret_1m

        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        month_ends = close.resample("BME").last().index

        for date in month_ends:
            if date not in close.index:
                continue

            factors = pd.DataFrame({
                "momentum": momentum_raw.loc[date],
                "low_vol": low_vol_raw.loc[date],
                "reversal": reversal_raw.loc[date],
            }).dropna()

            if len(factors) < 5:
                continue

            # Cross-sectional z-score each factor
            zscored = _cross_sectional_zscore(factors)

            # Combined score
            combined = (
                zscored["momentum"] * self.factor_weights["momentum"]
                + zscored["low_vol"] * self.factor_weights["low_vol"]
                + zscored["reversal"] * self.factor_weights["reversal"]
            )

            n = len(combined)
            quintile_size = max(1, n // 5)
            long_tickers = combined.nlargest(quintile_size).index
            short_tickers = combined.nsmallest(quintile_size).index

            long_weight = 0.5 / len(long_tickers)
            short_weight = -0.5 / len(short_tickers)

            signals.loc[date, :] = 0.0
            for t in long_tickers:
                if t in signals.columns:
                    signals.loc[date, t] = long_weight
            for t in short_tickers:
                if t in signals.columns:
                    signals.loc[date, t] = short_weight

        # Forward-fill between rebalance dates (NaN init preserves rebalance-date zeros).
        signals = signals.ffill().fillna(0.0)
        signals.iloc[:252] = 0.0

        return signals

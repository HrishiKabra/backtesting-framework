# tests/test_strategies.py
import pandas as pd
import numpy as np
import pytest
from strategies.base import Strategy
from core.engine import LookaheadBarrier


def test_strategy_is_abstract():
    """Cannot instantiate Strategy directly."""
    with pytest.raises(TypeError):
        Strategy()


class ConcreteStrategy(Strategy):
    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].fillna(0.0)
        return pd.DataFrame(0.05, index=close.index, columns=close.columns)


def test_concrete_strategy_returns_dataframe(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = ConcreteStrategy()
    signals = strategy.generate_signals(barrier)
    assert isinstance(signals, pd.DataFrame)
    assert signals.index.equals(synthetic_data.index)


def test_signals_contain_no_nans(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = ConcreteStrategy()
    signals = strategy.generate_signals(barrier)
    assert not signals.isnull().any().any()


# --- MomentumStrategy tests (Task 10) ---

from strategies.momentum import MomentumStrategy


def test_momentum_signals_shape(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MomentumStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert signals.shape == (len(synthetic_data), len(basic_config.tickers))
    assert not signals.isnull().any().any()


def test_momentum_warmup_period_is_zero(synthetic_data, basic_config):
    """First 252 rows must be 0 (warmup period — no 12-month history yet)."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MomentumStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals.iloc[:252] == 0.0).all().all()


def test_momentum_weights_bounded(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MomentumStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals >= -1.0).all().all()
    assert (signals <= 1.0).all().all()

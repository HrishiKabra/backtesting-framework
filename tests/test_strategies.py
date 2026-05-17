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

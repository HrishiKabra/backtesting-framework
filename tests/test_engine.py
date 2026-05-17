import pandas as pd
import numpy as np
import pytest
from core.engine import LookaheadBarrier, LookaheadBiasError


def test_get_history_excludes_signal_date(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    signal_date = synthetic_data.index[10]
    history = barrier.get_history(signal_date)
    assert history.index.max() < signal_date


def test_get_history_raises_on_date_not_in_index(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    bad_date = pd.Timestamp("2019-01-01")  # before data starts
    with pytest.raises(LookaheadBiasError):
        barrier.get_history(bad_date)


def test_get_shifted_data_lags_by_one(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    shifted = barrier.get_shifted_data()
    # Row at index[1] should equal the original row at index[0]
    original_row = synthetic_data.iloc[0]
    shifted_row = shifted.iloc[1]
    pd.testing.assert_series_equal(original_row, shifted_row, check_names=False)


def test_get_shifted_data_first_row_is_nan(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    shifted = barrier.get_shifted_data()
    assert shifted.iloc[0].isna().all()


# --- BacktestEngine tests (Task 8) ---

from config import BacktestConfig
from core.engine import BacktestEngine
from strategies.base import Strategy


class BuyAndHoldSPY(Strategy):
    """Trivial strategy: always 100% long SPY."""
    def generate_signals(self, barrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        if "SPY" in signals.columns:
            signals["SPY"] = 1.0
        return signals.fillna(0.0)


def test_engine_returns_nav_series(synthetic_data, basic_config):
    engine = BacktestEngine(basic_config, synthetic_data)
    strategy = BuyAndHoldSPY()
    nav = engine.run(strategy)
    assert isinstance(nav, pd.Series)
    assert len(nav) == len(synthetic_data)
    assert not nav.isnull().any()


def test_engine_nav_starts_near_initial_capital(synthetic_data, basic_config):
    engine = BacktestEngine(basic_config, synthetic_data)
    nav = engine.run(BuyAndHoldSPY())
    assert abs(nav.iloc[0] - basic_config.initial_capital) / basic_config.initial_capital < 0.02


def test_engine_signals_generated_once(synthetic_data, basic_config):
    """generate_signals should be called exactly once."""
    call_count = {"n": 0}
    class CountingStrategy(Strategy):
        def generate_signals(self, barrier):
            call_count["n"] += 1
            data = barrier.get_shifted_data()
            close = data["Close"].fillna(0.0)
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

    engine = BacktestEngine(basic_config, synthetic_data)
    engine.run(CountingStrategy())
    assert call_count["n"] == 1


def test_engine_raises_on_nan_signals(synthetic_data, basic_config):
    class NaNStrategy(Strategy):
        def generate_signals(self, barrier):
            data = barrier.get_shifted_data()
            return data["Close"]  # first row is NaN — not filled

    engine = BacktestEngine(basic_config, synthetic_data)
    with pytest.raises(AssertionError):
        engine.run(NaNStrategy())

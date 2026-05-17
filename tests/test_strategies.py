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


def test_momentum_accepts_custom_params(synthetic_data, basic_config):
    """params={'lookback_months': 6} → warmup is 126 days (6*21), not 252."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MomentumStrategy(basic_config, params={"lookback_months": 6, "skip_months": 1})
    assert strategy.lookback_days == 126
    assert strategy.skip_days == 21
    signals = strategy.generate_signals(barrier)
    assert (signals.iloc[:126] == 0.0).all().all()
    assert not signals.isnull().any().any()


def test_momentum_defaults_unchanged_without_params(basic_config):
    """No params → lookback_days=252, skip_days=21."""
    strategy = MomentumStrategy(basic_config)
    assert strategy.lookback_days == 252
    assert strategy.skip_days == 21


def test_momentum_has_param_grid():
    assert hasattr(MomentumStrategy, "param_grid")
    assert "lookback_months" in MomentumStrategy.param_grid
    assert "skip_months" in MomentumStrategy.param_grid


# --- BollingerStrategy tests (Task 11) ---

from strategies.mean_reversion import BollingerStrategy


def test_bollinger_signals_shape(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert signals.shape == (len(synthetic_data), len(basic_config.tickers))
    assert not signals.isnull().any().any()


def test_bollinger_warmup_is_zero(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    # First 20 rows have no rolling window — should be 0
    assert (signals.iloc[:20] == 0.0).all().all()


def test_bollinger_weights_bounded(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals.abs() <= 0.1).all().all()


def test_bollinger_generates_nonzero_signals(synthetic_data, basic_config):
    """Strategy must actually fire on 500 days of synthetic data."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals != 0.0).any().any()


def test_bollinger_accepts_custom_params(synthetic_data, basic_config):
    """params dict overrides default window and entry_z."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config, params={"window": 10, "entry_z": 1.5})
    assert strategy.window == 10
    assert strategy.entry_z == 1.5
    signals = strategy.generate_signals(barrier)
    assert not signals.isnull().any().any()


def test_bollinger_defaults_unchanged_without_params(synthetic_data, basic_config):
    """No params → window=20, entry_z=2.0 (existing behavior preserved)."""
    strategy = BollingerStrategy(basic_config)
    assert strategy.window == 20
    assert strategy.entry_z == 2.0


def test_bollinger_has_param_grid():
    assert hasattr(BollingerStrategy, "param_grid")
    assert "window" in BollingerStrategy.param_grid
    assert "entry_z" in BollingerStrategy.param_grid


# --- PairsTradingStrategy tests (Task 12) ---

from strategies.pairs_trading import PairsTradingStrategy


def test_pairs_signals_shape(synthetic_data, basic_config):
    pairs = [("AAPL", "MSFT")]
    barrier = LookaheadBarrier(synthetic_data)
    strategy = PairsTradingStrategy(basic_config, pairs=pairs)
    signals = strategy.generate_signals(barrier)
    assert signals.shape[0] == len(synthetic_data)
    assert not signals.isnull().any().any()


def test_pairs_training_period_is_zero(synthetic_data, basic_config):
    """First 252 rows must be 0 — training period."""
    pairs = [("AAPL", "MSFT")]
    barrier = LookaheadBarrier(synthetic_data)
    strategy = PairsTradingStrategy(basic_config, pairs=pairs, training_days=252)
    signals = strategy.generate_signals(barrier)
    assert (signals.iloc[:252] == 0.0).all().all()


def test_pairs_hedge_ratio_baked_in(synthetic_data, basic_config):
    """When long/short spread, A and B weights must have opposite signs."""
    # Construct explicitly cointegrated pair: AAPL = 1.5 * MSFT + noise
    rng = np.random.default_rng(42)
    n = len(synthetic_data)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = 1.5 * x + rng.normal(0, 0.5, n)

    coint_data = synthetic_data.copy()
    coint_data[("Close", "AAPL")] = y
    coint_data[("Close", "MSFT")] = x

    pairs = [("AAPL", "MSFT")]
    barrier = LookaheadBarrier(coint_data)
    strategy = PairsTradingStrategy(basic_config, pairs=pairs, training_days=100)
    signals = strategy.generate_signals(barrier)

    active = signals[(signals != 0).any(axis=1)]
    assert len(active) > 0, "Strategy should generate signals on cointegrated data"
    row = active.iloc[0]
    aapl_w = row.get("AAPL", 0.0)
    msft_w = row.get("MSFT", 0.0)
    assert aapl_w != 0.0 and msft_w != 0.0
    assert np.sign(aapl_w) != np.sign(msft_w)


def test_pairs_accepts_custom_params(basic_config):
    """params dict correctly overrides entry_z and exit_z."""
    strategy = PairsTradingStrategy(basic_config, params={"entry_z": 1.5, "exit_z": 0.5})
    assert strategy.z_entry == 1.5
    assert strategy.z_exit == 0.5


def test_pairs_defaults_unchanged_without_params(basic_config):
    """PairsTradingStrategy(config) uses z_entry=2.0, z_exit=0.0 defaults."""
    strategy = PairsTradingStrategy(basic_config)
    assert strategy.z_entry == 2.0
    assert strategy.z_exit == 0.0


def test_pairs_has_param_grid():
    """param_grid class attribute exists with correct keys."""
    assert hasattr(PairsTradingStrategy, "param_grid")
    assert "entry_z" in PairsTradingStrategy.param_grid
    assert "exit_z" in PairsTradingStrategy.param_grid


# --- MultiFactorStrategy tests (Task 13) ---

from strategies.multi_factor import MultiFactorStrategy


def test_multifactor_signals_shape(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MultiFactorStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert signals.shape == (len(synthetic_data), len(basic_config.tickers))
    assert not signals.isnull().any().any()


def test_multifactor_warmup_is_zero(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MultiFactorStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals.iloc[:252] == 0.0).all().all()


def test_multifactor_weights_bounded(synthetic_data, basic_config):
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MultiFactorStrategy(basic_config)
    signals = strategy.generate_signals(barrier)
    assert (signals >= -1.0).all().all()
    assert (signals <= 1.0).all().all()

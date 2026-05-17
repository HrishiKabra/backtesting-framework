import itertools
import pandas as pd
import numpy as np
import pytest
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class TinyStrategy(Strategy):
    """Minimal strategy for fast WFO tests — no computation, instant signals."""
    param_grid = {"weight": [0.05, 0.1]}

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        self.weight = (params or {}).get("weight", 0.05)

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].fillna(0.0)
        return pd.DataFrame(self.weight, index=close.index, columns=close.columns)


def test_wfo_returns_dataframe(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert isinstance(results, pd.DataFrame)


def test_wfo_result_has_required_columns(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert "weight" in results.columns
    assert "in_sample_sharpe" in results.columns
    assert "out_sample_sharpe" in results.columns
    assert "degradation" in results.columns


def test_wfo_one_row_per_combo(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert len(results) == 2  # TinyStrategy has 2 combos: weight=[0.05, 0.1]


def test_wfo_sorted_by_oos_sharpe(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    oos = results["out_sample_sharpe"].values
    assert list(oos) == sorted(oos, reverse=True)


def test_wfo_degradation_equals_diff(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    for _, row in results.iterrows():
        expected = row["in_sample_sharpe"] - row["out_sample_sharpe"]
        assert abs(row["degradation"] - expected) < 1e-9


def test_wfo_filters_invalid_multifactor_combos():
    from core.walk_forward import WalkForwardOptimizer
    combos = [
        {"momentum_wt": 0.7, "lowvol_wt": 0.5},  # 1.2 → invalid
        {"momentum_wt": 0.7, "lowvol_wt": 0.3},  # 1.0 → valid
        {"momentum_wt": 0.5, "lowvol_wt": 0.3},  # 0.8 → valid
    ]
    filtered = WalkForwardOptimizer._filter_combos(combos)
    assert len(filtered) == 2
    assert not any(
        c["momentum_wt"] + c["lowvol_wt"] > 1.0
        for c in filtered
    )


def test_wfo_base_params_merged_into_combo(synthetic_data, basic_config):
    """base_params are passed to every strategy instantiation alongside grid combo."""
    received = []

    class RecordingStrategy(Strategy):
        param_grid = {"weight": [0.05]}

        def __init__(self, config, params=None):
            self.config = config
            received.append(dict(params or {}))

        def generate_signals(self, barrier):
            data = barrier.get_shifted_data()
            close = data["Close"].fillna(0.0)
            return pd.DataFrame(0.05, index=close.index, columns=close.columns)

    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(
        RecordingStrategy, synthetic_data, basic_config,
        base_params={"fixed_key": "fixed_val"}
    )
    optimizer.run(train_years=1, test_months=3)
    assert all("fixed_key" in p for p in received)
    assert all(p["fixed_key"] == "fixed_val" for p in received)

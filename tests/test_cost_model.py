# tests/test_cost_model.py
import numpy as np
import pytest
from config import BacktestConfig
from core.cost_model import CostModel


@pytest.fixture
def model(basic_config):
    return CostModel(basic_config)


def test_buy_fill_price_above_close(model):
    fill = model.get_fill_price(close_price=100.0, order_shares=10, avg_volume_notional=1e9)
    assert fill > 100.0


def test_sell_fill_price_below_close(model):
    fill = model.get_fill_price(close_price=100.0, order_shares=-10, avg_volume_notional=1e9)
    assert fill < 100.0


def test_market_impact_applied_for_large_order(model):
    # Small order: no market impact
    np.random.seed(42)
    fill_small = model.get_fill_price(100.0, order_shares=10, avg_volume_notional=1e9)
    # Large order: exceeds 0.5% of avg_volume_notional
    # 0.5% of 1e9 = 5,000,000 notional threshold
    # order of 100,000 shares @ $100 = $10,000,000 notional -> triggers impact
    np.random.seed(42)
    fill_large = model.get_fill_price(100.0, order_shares=100_000, avg_volume_notional=1e9)
    # With same random seed, large order should have higher fill due to market impact
    assert fill_large > fill_small


def test_market_impact_not_applied_for_small_order(model):
    # Order of 10 shares @ $100 = $1,000 notional, well under 0.5% of 1e9
    np.random.seed(0)
    fill = model.get_fill_price(100.0, order_shares=10, avg_volume_notional=1e9)
    # Fill should only include spread (0.05%) + slippage, no market impact
    # Max expected: 100 * (1 + 0.0005) * (1 + 0.001) ≈ 100.15
    assert fill < 100.20


def test_commission_flat_plus_pct(model):
    commission = model.get_commission(order_shares=100, fill_price=50.0)
    # 100 shares * $50 = $5,000 notional
    # flat $1 + 0.01% * $5,000 = $1 + $0.50 = $1.50
    assert abs(commission - 1.50) < 0.01


def test_slippage_always_adversarial(basic_config):
    """Slippage should always increase cost, never decrease it."""
    np.random.seed(99)
    model = CostModel(basic_config)
    # Run many trials; fill should always be above close for buys
    for _ in range(50):
        fill = model.get_fill_price(100.0, order_shares=1, avg_volume_notional=1e9)
        # Minimum is close * (1 + half_spread) = 100.05; slippage adds more
        assert fill >= 100.0 * (1 + basic_config.half_spread)

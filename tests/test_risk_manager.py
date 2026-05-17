# tests/test_risk_manager.py
import pandas as pd
import numpy as np
import pytest
from config import BacktestConfig
from core.portfolio import Portfolio
from core.risk_manager import RiskManager


@pytest.fixture
def config():
    return BacktestConfig(
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        initial_capital=100_000.0,
        max_position_pct=0.20,
    )


@pytest.fixture
def risk_manager(config):
    return RiskManager(config)


@pytest.fixture
def empty_portfolio(config):
    return Portfolio(config.initial_capital)


def test_small_order_approved(risk_manager, empty_portfolio):
    orders = pd.Series({"AAPL": 10})
    close = pd.Series({"AAPL": 100.0})  # 10*100 = $1,000 = 1% of $100k NAV
    approved = risk_manager.validate(orders, empty_portfolio, close)
    assert approved["AAPL"] == 10


def test_oversized_order_capped(risk_manager, empty_portfolio):
    # 300 shares @ $100 = $30,000 = 30% of $100k NAV → exceeds 20% limit
    orders = pd.Series({"AAPL": 300})
    close = pd.Series({"AAPL": 100.0})
    approved = risk_manager.validate(orders, empty_portfolio, close)
    # Max allowed: 20% of $100k / $100 = 200 shares
    assert approved["AAPL"] <= 200


def test_short_margin_rejection(risk_manager, empty_portfolio):
    # Shorting 400 shares @ $300 = $120,000 notional
    # Required margin = 1.5 * $120,000 = $180,000 > $100,000 cash → reject
    orders = pd.Series({"SPY": -400})
    close = pd.Series({"SPY": 300.0})
    approved = risk_manager.validate(orders, empty_portfolio, close)
    assert approved.get("SPY", 0) == 0


def test_short_within_margin_approved(risk_manager, empty_portfolio):
    # Shorting 100 shares @ $100 = $10,000 notional
    # Required margin = 1.5 * $10,000 = $15,000 < $100,000 cash → approve
    orders = pd.Series({"AAPL": -100})
    close = pd.Series({"AAPL": 100.0})
    approved = risk_manager.validate(orders, empty_portfolio, close)
    assert approved.get("AAPL", 0) == -100

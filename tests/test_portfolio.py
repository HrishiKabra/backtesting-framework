# tests/test_portfolio.py
import pandas as pd
import pytest
from core.portfolio import Portfolio


@pytest.fixture
def portfolio():
    return Portfolio(initial_capital=100_000.0)


def test_initial_nav_equals_capital(portfolio):
    assert portfolio.nav == 100_000.0


def test_buy_reduces_cash(portfolio):
    portfolio.execute_order("AAPL", shares=10, fill_price=150.0, commission=2.0,
                            date=pd.Timestamp("2020-01-02"))
    # cost = 10 * 150 + 2 = 1502
    assert abs(portfolio.cash - (100_000.0 - 1502.0)) < 0.01


def test_sell_increases_cash(portfolio):
    # First buy
    portfolio.execute_order("AAPL", shares=10, fill_price=150.0, commission=2.0,
                            date=pd.Timestamp("2020-01-02"))
    cash_after_buy = portfolio.cash
    # Then sell all
    portfolio.execute_order("AAPL", shares=-10, fill_price=155.0, commission=2.0,
                            date=pd.Timestamp("2020-01-03"))
    # proceeds = 10 * 155 - 2 = 1548
    assert abs(portfolio.cash - (cash_after_buy + 1548.0)) < 0.01


def test_positions_updated_correctly(portfolio):
    portfolio.execute_order("AAPL", shares=5, fill_price=100.0, commission=1.0,
                            date=pd.Timestamp("2020-01-02"))
    assert portfolio.positions["AAPL"] == 5.0

    portfolio.execute_order("AAPL", shares=-5, fill_price=105.0, commission=1.0,
                            date=pd.Timestamp("2020-01-03"))
    assert "AAPL" not in portfolio.positions


def test_mark_to_market_updates_nav(portfolio):
    portfolio.execute_order("AAPL", shares=10, fill_price=100.0, commission=1.0,
                            date=pd.Timestamp("2020-01-02"))
    close_prices = pd.Series({"AAPL": 110.0})
    portfolio.mark_to_market(close_prices, pd.Timestamp("2020-01-02"))
    # cash = 100000 - (10*100 + 1) = 98999; position = 10*110 = 1100
    expected_nav = portfolio.cash + 10 * 110.0
    assert abs(portfolio.nav - expected_nav) < 0.01


def test_nav_history_recorded(portfolio):
    close = pd.Series({"AAPL": 100.0})
    portfolio.mark_to_market(close, pd.Timestamp("2020-01-02"))
    portfolio.mark_to_market(close, pd.Timestamp("2020-01-03"))
    assert len(portfolio.nav_history) == 2


def test_transaction_log_recorded(portfolio):
    portfolio.execute_order("MSFT", shares=3, fill_price=200.0, commission=1.5,
                            date=pd.Timestamp("2020-01-02"))
    assert len(portfolio.transaction_log) == 1
    tx = portfolio.transaction_log[0]
    assert tx.ticker == "MSFT"
    assert tx.shares == 3
    assert tx.side == "buy"


def test_short_position_tracked(portfolio):
    portfolio.execute_order("SPY", shares=-20, fill_price=300.0, commission=2.0,
                            date=pd.Timestamp("2020-01-02"))
    assert portfolio.positions.get("SPY", 0) == -20.0
    # cash increases from short sale proceeds minus commission
    assert portfolio.cash > 100_000.0

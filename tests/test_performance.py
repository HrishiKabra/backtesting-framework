import pandas as pd
import numpy as np
import pytest
from core.performance import PerformanceAnalyzer


@pytest.fixture
def flat_nav():
    """NAV that grows at exactly 10% per year — known Sharpe."""
    dates = pd.bdate_range("2020-01-02", periods=252)
    daily_return = (1.10) ** (1 / 252) - 1
    nav = pd.Series(
        100_000 * np.cumprod(np.ones(252) * (1 + daily_return)),
        index=dates,
    )
    return nav


@pytest.fixture
def volatile_nav():
    """NAV with known random walk."""
    np.random.seed(7)
    dates = pd.bdate_range("2020-01-02", periods=500)
    rets = np.random.normal(0.0003, 0.015, 500)
    nav = pd.Series(100_000 * np.cumprod(1 + rets), index=dates)
    return nav


def test_annualized_return(flat_nav):
    pa = PerformanceAnalyzer(flat_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    assert abs(metrics["annualized_return"] - 0.10) < 0.001


def test_sharpe_positive_for_good_strategy(flat_nav):
    pa = PerformanceAnalyzer(flat_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    # 10% return, 5% RFR → positive Sharpe
    assert metrics["sharpe_ratio"] > 0


def test_max_drawdown_negative(volatile_nav):
    pa = PerformanceAnalyzer(volatile_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    assert metrics["max_drawdown"] < 0


def test_sortino_has_floor(flat_nav):
    """With no negative return days, downside_vol uses floor — no division by zero."""
    pa = PerformanceAnalyzer(flat_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    assert np.isfinite(metrics["sortino_ratio"])


def test_monthly_returns_matrix_shape(volatile_nav):
    pa = PerformanceAnalyzer(volatile_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    matrix = metrics["monthly_returns_matrix"]
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[1] == 12  # 12 months as columns


def test_hit_rate_between_0_and_1(volatile_nav):
    pa = PerformanceAnalyzer(volatile_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    assert 0 <= metrics["hit_rate"] <= 1


def test_calmar_positive_when_positive_return(flat_nav):
    pa = PerformanceAnalyzer(flat_nav, risk_free_rate=0.0)
    metrics = pa.compute()
    assert metrics["calmar_ratio"] > 0

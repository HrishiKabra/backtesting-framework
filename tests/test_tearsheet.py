# tests/test_tearsheet.py
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from core.performance import PerformanceAnalyzer
from reporting.tearsheet import TearsheetGenerator


@pytest.fixture
def sample_nav():
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    rets = np.random.normal(0.0004, 0.012, 500)
    return pd.Series(100_000 * np.cumprod(1 + rets), index=dates)


@pytest.fixture
def sample_spy_nav():
    np.random.seed(99)
    dates = pd.bdate_range("2020-01-02", periods=500)
    rets = np.random.normal(0.0003, 0.010, 500)
    return pd.Series(100_000 * np.cumprod(1 + rets), index=dates)


def test_tearsheet_creates_png(sample_nav, sample_spy_nav):
    pa = PerformanceAnalyzer(sample_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "tearsheet.png")
        gen = TearsheetGenerator(strategy_name="Test")
        gen.render(metrics, sample_nav, sample_spy_nav, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 10_000  # non-trivial file


def test_tearsheet_spy_same_length(sample_nav):
    """SPY benchmark must align with strategy NAV — same index."""
    pa = PerformanceAnalyzer(sample_nav, risk_free_rate=0.05)
    metrics = pa.compute()
    spy_wrong = sample_nav.iloc[:-10]  # misaligned — shorter
    gen = TearsheetGenerator(strategy_name="Test")
    with pytest.raises(ValueError, match="must share the same index"):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.render(metrics, sample_nav, spy_wrong, os.path.join(tmpdir, "t.png"))

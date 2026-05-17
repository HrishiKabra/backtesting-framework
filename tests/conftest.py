import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_data():
    """500 business days of synthetic OHLCV for AAPL, MSFT, SPY."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    tickers = ["AAPL", "MSFT", "SPY"]

    series_list = []
    for ticker in tickers:
        rets = np.random.normal(0.0003, 0.015, len(dates))
        close = 100.0 * np.cumprod(1 + rets)
        volume = np.random.randint(10_000_000, 100_000_000, len(dates)).astype(float)
        for field, vals in [
            ("Open", close * 0.999),
            ("High", close * 1.005),
            ("Low", close * 0.995),
            ("Close", close),
            ("Volume", volume),
        ]:
            series_list.append(
                pd.Series(vals, index=dates, name=(field, ticker))
            )

    df = pd.concat(series_list, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def basic_config():
    from config import BacktestConfig
    return BacktestConfig(
        tickers=["AAPL", "MSFT", "SPY"],
        start_date="2020-01-02",
        end_date="2021-12-31",
        initial_capital=100_000.0,
    )

# tests/test_data_loader.py
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest
from config import BacktestConfig
from core.data_loader import DataLoader


@pytest.fixture
def config():
    return BacktestConfig(
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-02",
        end_date="2020-12-31",
    )


def _make_fake_yf_data(tickers, dates):
    """Produces the same MultiIndex structure yfinance returns."""
    series_list = []
    for ticker in tickers:
        close = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
        vol = np.random.randint(10_000_000, 50_000_000, len(dates)).astype(float)
        for field, vals in [
            ("Open", close), ("High", close * 1.01), ("Low", close * 0.99),
            ("Close", close), ("Volume", vol),
        ]:
            series_list.append(pd.Series(vals, index=dates, name=(field, ticker)))
    df = pd.concat(series_list, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_fetch_returns_multiindex_dataframe(config):
    dates = pd.bdate_range(config.start_date, config.end_date)
    fake_data = _make_fake_yf_data(config.tickers, dates)

    with patch("core.data_loader.yf.download", return_value=fake_data):
        loader = DataLoader(config)
        data = loader.fetch()

    assert isinstance(data.columns, pd.MultiIndex)
    assert "Close" in data.columns.get_level_values(0)
    assert "Volume" in data.columns.get_level_values(0)
    assert set(config.tickers) == set(data.columns.get_level_values(1).unique())


def test_fetch_forward_fills_missing(config):
    dates = pd.bdate_range(config.start_date, config.end_date)
    fake_data = _make_fake_yf_data(config.tickers, dates)
    # Inject NaN in the middle
    fake_data.loc[dates[5], ("Close", "AAPL")] = np.nan

    with patch("core.data_loader.yf.download", return_value=fake_data):
        loader = DataLoader(config)
        data = loader.fetch()

    assert not data["Close"]["AAPL"].isna().any()


def test_fetch_warns_on_excessive_missing(config, capsys):
    dates = pd.bdate_range(config.start_date, config.end_date)
    fake_data = _make_fake_yf_data(config.tickers, dates)
    # Blank out 10% of AAPL close (before forward fill — set to NaN at yf level)
    n_missing = int(len(dates) * 0.10)
    missing_idx = dates[:n_missing]
    for i in missing_idx:
        fake_data.loc[i, ("Close", "AAPL")] = np.nan

    with patch("core.data_loader.yf.download", return_value=fake_data):
        loader = DataLoader(config)
        loader.fetch()

    captured = capsys.readouterr()
    assert "AAPL" in captured.out  # warning printed

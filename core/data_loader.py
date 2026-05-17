import yfinance as yf
import pandas as pd
from config import BacktestConfig


class DataLoader:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def fetch(self) -> pd.DataFrame:
        """
        Downloads OHLCV data from yfinance, forward-fills gaps, and validates
        data quality. Returns a DataFrame with MultiIndex columns (field, ticker).
        """
        data = yf.download(
            tickers=self.config.tickers,
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=True,
            progress=False,
        )

        # Ensure MultiIndex columns even for single-ticker downloads
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_tuples(
                [(col, self.config.tickers[0]) for col in data.columns]
            )

        self._warn_missing(data)
        data = data.ffill()
        return data

    def _warn_missing(self, data: pd.DataFrame) -> None:
        """Prints a warning for any ticker with >5% missing close prices."""
        close = data["Close"]
        n_days = len(data)
        for ticker in close.columns:
            pct_missing = close[ticker].isna().sum() / n_days
            if pct_missing > 0.05:
                print(
                    f"WARNING: {ticker} has {pct_missing:.1%} missing close prices "
                    f"({int(pct_missing * n_days)} of {n_days} days)"
                )

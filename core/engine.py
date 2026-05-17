import pandas as pd


class LookaheadBiasError(Exception):
    """Raised when a strategy attempts to access data from the future."""


class LookaheadBarrier:
    """
    Wraps the raw OHLCV DataFrame and enforces that strategies can only access
    data strictly before a given signal date. This is structural prevention of
    lookahead bias — enforced at the API level, not by convention.
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data

    def get_history(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Returns all rows with index strictly before `as_of`."""
        if as_of not in self._data.index:
            raise LookaheadBiasError(
                f"{as_of} is not in the data index. "
                "Signal dates must be trading days within the loaded data range."
            )
        return self._data.loc[self._data.index < as_of]

    def get_shifted_data(self) -> pd.DataFrame:
        """
        Returns the full data shifted forward by 1 trading day.
        Value at index T reflects data available at market close of T-1.
        Use this for fully vectorized signal generation.
        """
        return self._data.shift(1)


import numpy as np
from config import BacktestConfig
from core.cost_model import CostModel
from core.portfolio import Portfolio
from core.risk_manager import RiskManager


class BacktestEngine:
    def __init__(self, config: BacktestConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.trading_dates = data.index
        self.barrier = LookaheadBarrier(data)
        self.cost_model = CostModel(config)
        self.portfolio = Portfolio(config.initial_capital)
        self.risk_manager = RiskManager(config)
        # Pre-compute 20-day avg volume x price (notional). Shifted so T uses T-1 data.
        self.avg_volume_notional = (
            (data["Volume"] * data["Close"])
            .rolling(config.avg_volume_window)
            .mean()
            .shift(1)
        )

    def run(self, strategy) -> pd.Series:
        """
        Runs the backtest. Calls strategy.generate_signals() once before the loop,
        then iterates day-by-day for execution only.
        """
        signals_df = strategy.generate_signals(self.barrier)

        assert not signals_df.isnull().any().any(), (
            "Signals contain NaN values. Strategies must fill training periods with 0.0."
        )
        assert signals_df.index.equals(self.trading_dates), (
            "Signals index does not match trading dates."
        )

        nav_series = pd.Series(index=self.trading_dates, dtype=float)

        for date in self.trading_dates:
            close_prices = self.data["Close"].loc[date]
            signals_today = signals_df.loc[date]

            orders = self._signals_to_orders(signals_today, close_prices)
            approved = self.risk_manager.validate(orders, self.portfolio, close_prices)

            avg_vol_row = self.avg_volume_notional.loc[date]

            for ticker, order_shares in approved.items():
                if abs(order_shares) < 1e-6:
                    continue
                avg_vol = avg_vol_row.get(ticker, 0.0)
                if pd.isna(avg_vol):
                    avg_vol = 0.0
                fill_price = self.cost_model.get_fill_price(
                    close_price=float(close_prices.get(ticker, 0.0)),
                    order_shares=order_shares,
                    avg_volume_notional=float(avg_vol),
                )
                commission = self.cost_model.get_commission(order_shares, fill_price)
                self.portfolio.execute_order(ticker, order_shares, fill_price,
                                             commission, date)

            self.portfolio.mark_to_market(close_prices, date)
            nav_series[date] = self.portfolio.nav

        return nav_series

    def _signals_to_orders(self, signals: pd.Series, close_prices: pd.Series) -> pd.Series:
        """Convert target weight signals to share orders."""
        current_nav = self.portfolio.nav
        orders = {}
        for ticker, target_weight in signals.items():
            if ticker not in close_prices.index:
                continue
            price = close_prices[ticker]
            if pd.isna(price) or price <= 0:
                continue
            target_shares = (target_weight * current_nav) / price
            current_shares = self.portfolio.positions.get(ticker, 0.0)
            order_shares = target_shares - current_shares
            orders[ticker] = order_shares
        return pd.Series(orders, dtype=float)

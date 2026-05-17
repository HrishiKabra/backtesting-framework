import pandas as pd
from config import BacktestConfig
from core.portfolio import Portfolio


class RiskManager:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def validate(
        self,
        orders: pd.Series,          # ticker → order_shares
        portfolio: Portfolio,
        close_prices: pd.Series,    # ticker → close_price
    ) -> pd.Series:
        """
        Returns approved orders with oversized positions capped and margin
        violations zeroed out.
        """
        approved = {}
        current_nav = portfolio.nav

        for ticker, order_shares in orders.items():
            if abs(order_shares) < 1e-6:
                continue

            price = close_prices.get(ticker, 0.0)
            if price <= 0:
                continue

            current_shares = portfolio.positions.get(ticker, 0.0)
            target_shares = current_shares + order_shares

            # Margin check for shorts (before position cap): require 150% of short notional in cash
            if target_shares < 0:
                required_margin = 1.5 * abs(target_shares) * price
                if required_margin > portfolio.cash:
                    continue  # reject the order

            target_notional = abs(target_shares) * price

            # Cap position at max_position_pct of NAV
            max_notional = self.config.max_position_pct * current_nav
            if target_notional > max_notional and current_nav > 0:
                max_shares = max_notional / price
                capped_target = max_shares if target_shares > 0 else -max_shares
                order_shares = capped_target - current_shares

            if abs(order_shares) > 1e-6:
                approved[ticker] = order_shares

        return pd.Series(approved, dtype=float)

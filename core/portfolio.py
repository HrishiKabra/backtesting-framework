from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd


@dataclass
class Transaction:
    date: pd.Timestamp
    ticker: str
    shares: float       # positive = buy, negative = sell/short
    fill_price: float
    commission: float
    side: str           # "buy" or "sell"


class Portfolio:
    def __init__(self, initial_capital: float):
        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {}           # ticker → shares
        self.nav_history: List[Tuple[pd.Timestamp, float]] = []
        self.transaction_log: List[Transaction] = []
        self._last_prices: pd.Series = pd.Series(dtype=float)

    @property
    def nav(self) -> float:
        position_value = sum(
            shares * self._last_prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + position_value

    def execute_order(
        self,
        ticker: str,
        shares: float,
        fill_price: float,
        commission: float,
        date: pd.Timestamp,
    ) -> None:
        notional = abs(shares) * fill_price
        if shares > 0:  # buy
            self.cash -= notional + commission
        else:           # sell / cover short
            self.cash += notional - commission

        current = self.positions.get(ticker, 0.0)
        new_position = current + shares
        if abs(new_position) < 1e-6:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = new_position

        self.transaction_log.append(Transaction(
            date=date,
            ticker=ticker,
            shares=shares,
            fill_price=fill_price,
            commission=commission,
            side="buy" if shares > 0 else "sell",
        ))

    def mark_to_market(self, close_prices: pd.Series, date: pd.Timestamp) -> None:
        """Updates last known prices and records NAV. Call once per trading day."""
        self._last_prices = close_prices
        self.nav_history.append((date, self.nav))

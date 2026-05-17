import numpy as np
from config import BacktestConfig


class CostModel:
    """
    Computes all-in fill price and commission for each order.

    # NOTE: using close as fill price proxy — next-day open execution would add
    # ~0.1-0.3% additional slippage not captured here.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def get_fill_price(
        self,
        close_price: float,
        order_shares: float,
        avg_volume_notional: float,
    ) -> float:
        """
        Returns all-in fill price including bid-ask spread, market impact,
        and adversarial slippage.
        """
        is_buy = order_shares > 0

        # 1. Bid-ask spread
        if is_buy:
            price = close_price * (1 + self.config.half_spread)
        else:
            price = close_price * (1 - self.config.half_spread)

        # 2. Market impact (only for large orders)
        order_notional = abs(order_shares) * close_price
        if avg_volume_notional > 0:
            threshold = self.config.market_impact_threshold * avg_volume_notional
            if order_notional > threshold:
                impact = self.config.market_impact_coeff * np.sqrt(
                    order_notional / avg_volume_notional
                )
                price *= (1 + impact) if is_buy else (1 - impact)

        # 3. Adversarial slippage: abs() ensures it always increases cost
        slippage = abs(np.random.normal(0, self.config.slippage_std))
        price *= (1 + slippage) if is_buy else (1 - slippage)

        return price

    def get_commission(self, order_shares: float, fill_price: float) -> float:
        """Flat $1 + 0.01% of notional."""
        notional = abs(order_shares) * fill_price
        return self.config.commission_flat + self.config.commission_pct * notional

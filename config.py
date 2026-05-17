from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    half_spread: float = 0.0005        # 0.05% — liquid large-caps
    commission_flat: float = 1.0       # $1 per trade
    commission_pct: float = 0.0001     # 0.01% of notional
    slippage_std: float = 0.0002       # 0.02% std — adversarial draw
    market_impact_threshold: float = 0.005   # 0.5% of avg daily volume
    market_impact_coeff: float = 0.001       # 0.1% * sqrt(order/volume)
    max_position_pct: float = 0.20     # 20% of NAV per ticker
    max_leverage: float = 1.5          # abs sum of weights
    stop_loss_pct: float = -0.15       # -15% from entry (portfolio-level)
    risk_free_rate: float = 0.05       # 5% annualized
    avg_volume_window: int = 20        # days for avg volume calculation

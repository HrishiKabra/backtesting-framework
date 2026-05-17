"""
run_backtest.py — entry point for the backtesting framework.

Usage:
    python run_backtest.py --strategy momentum
    python run_backtest.py --strategy bollinger
    python run_backtest.py --strategy pairs
    python run_backtest.py --strategy multifactor
    python run_backtest.py --validate   # SPY buy-and-hold validation
"""
import argparse
import os
import pandas as pd
import numpy as np

from config import BacktestConfig
from universe import DEV_UNIVERSE
from core.data_loader import DataLoader
from core.engine import BacktestEngine
from core.performance import PerformanceAnalyzer
from reporting.tearsheet import TearsheetGenerator
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import BollingerStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.multi_factor import MultiFactorStrategy
from strategies.base import Strategy
from core.engine import LookaheadBarrier


class BuyAndHoldSPY(Strategy):
    """
    Reference strategy: 100% long SPY at all times.

    NOTE: Not used by run_validation() because running this through
    BacktestEngine causes daily rebalancing — target_shares = 1.0 * NAV / price
    changes every day as NAV moves, generating orders on every bar. That is
    constant-weight rebalancing (not buy-and-hold) and produces rebalancing
    alpha in trending markets. run_validation() uses Portfolio + CostModel
    directly for a true single-purchase test.
    """
    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        if "SPY" in signals.columns:
            signals["SPY"] = 1.0
        return signals.fillna(0.0)


STRATEGIES = {
    "momentum": MomentumStrategy,
    "bollinger": BollingerStrategy,
    "pairs": PairsTradingStrategy,
    "multifactor": MultiFactorStrategy,
}


def run_validation(config: BacktestConfig) -> None:
    """
    SPY buy-and-hold validation gate.

    The engine uses target-weight signals, which cause daily rebalancing.
    For a single-asset 100% SPY signal, this produces 'rebalancing alpha'
    (the constant-weight rebalancing effect), causing the engine NAV to
    diverge from SPY's raw price return even with zero costs.

    To validate NAV accounting correctly, we instead:
    1. Manually buy SPY on day 1 using the Portfolio + CostModel directly
       (one purchase, no subsequent rebalancing).
    2. Mark-to-market each subsequent day using the Portfolio.
    3. Compare the resulting NAV to SPY's raw price return.

    A gap > 0.5% indicates a bug in Portfolio.execute_order or mark_to_market.
    """
    from core.cost_model import CostModel
    from core.portfolio import Portfolio

    print("\n=== SPY Buy-and-Hold Validation ===")
    val_config = BacktestConfig(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
        half_spread=config.half_spread,
        commission_flat=config.commission_flat,
        commission_pct=config.commission_pct,
        slippage_std=0.0,           # no random slippage for deterministic validation
        market_impact_threshold=config.market_impact_threshold,
        market_impact_coeff=config.market_impact_coeff,
        max_position_pct=1.0,
        max_leverage=1.5,
        stop_loss_pct=config.stop_loss_pct,
        risk_free_rate=config.risk_free_rate,
        avg_volume_window=config.avg_volume_window,
    )
    loader = DataLoader(val_config)
    data = loader.fetch()

    spy_close = data["Close"]["SPY"]
    spy_actual_return = spy_close.iloc[-1] / spy_close.iloc[0] - 1

    cost_model = CostModel(val_config)
    portfolio = Portfolio(val_config.initial_capital)
    nav_series = pd.Series(index=data.index, dtype=float)

    for i, date in enumerate(data.index):
        close_prices = data["Close"].loc[date]
        spy_price = float(close_prices.get("SPY", 0.0))
        if spy_price <= 0 or pd.isna(spy_price):
            portfolio.mark_to_market(close_prices, date)
            nav_series[date] = portfolio.nav
            continue

        if i == 0:
            # Day 1: buy as many SPY shares as possible with full capital
            shares_to_buy = val_config.initial_capital / spy_price
            avg_vol_notional = 0.0  # NaN on day 1 (no 20-day history)
            fill_price = cost_model.get_fill_price(spy_price, shares_to_buy, avg_vol_notional)
            commission = cost_model.get_commission(shares_to_buy, fill_price)
            portfolio.execute_order("SPY", shares_to_buy, fill_price, commission, date)

        portfolio.mark_to_market(close_prices, date)
        nav_series[date] = portfolio.nav

    strategy_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    gap = abs(strategy_return - spy_actual_return)

    print(f"SPY actual return:    {spy_actual_return:.4%}")
    print(f"Engine return:        {strategy_return:.4%}")
    print(f"Gap (after costs):    {gap:.4%}")

    if gap > 0.005:
        print(f"FAIL: gap {gap:.4%} exceeds 0.5% threshold — check cost model or NAV accounting.")
    else:
        print("PASS: engine correctly tracks SPY return within 0.5% after costs.")


def run_strategy(strategy_name: str, config: BacktestConfig) -> None:
    print(f"\n=== Running {strategy_name} strategy ===")
    loader = DataLoader(config)
    data = loader.fetch()

    strategy_cls = STRATEGIES[strategy_name]
    strategy = strategy_cls(config)

    engine = BacktestEngine(config, data)
    nav = engine.run(strategy)

    # SPY benchmark (same date range, normalized)
    spy_nav = data["Close"]["SPY"] if "SPY" in data["Close"].columns else nav
    spy_nav = spy_nav.reindex(nav.index).ffill()

    pa = PerformanceAnalyzer(nav, risk_free_rate=config.risk_free_rate)
    metrics = pa.compute()

    print("\nPerformance Metrics:")
    for key, val in metrics.items():
        if key == "monthly_returns_matrix":
            continue
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.4f}")
        else:
            print(f"  {key:35s}: {val}")

    os.makedirs("output", exist_ok=True)
    output_path = f"output/{strategy_name}_tearsheet.png"
    gen = TearsheetGenerator(strategy_name=strategy_name.title())
    gen.render(metrics, nav, spy_nav, output_path)
    print(f"\nTearsheet: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting Framework")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), help="Strategy to run")
    parser.add_argument("--validate", action="store_true", help="Run SPY buy-and-hold validation")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    cfg = BacktestConfig(
        tickers=DEV_UNIVERSE,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    if args.validate:
        run_validation(cfg)
    elif args.strategy:
        run_strategy(args.strategy, cfg)
    else:
        parser.print_help()

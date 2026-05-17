# Backtesting Framework

A vectorized, hybrid backtesting engine built from first principles in Python — no backtrader, zipline, or other framework. Implements four trading strategies with realistic market simulation.

---

## Architecture

Signal generation (vectorized, one upfront pass) is separated from execution (event-driven, per-day loop):

```
# Before the loop — strategy computes all signals at once
signals_df = strategy.generate_signals(barrier)   # full DataFrame, all dates × all tickers

# Inside the loop — execution only, no strategy calls
for each date T:
    signals_today = signals_df.loc[T]             # row read, no computation
    orders = risk_manager.validate(signals_today, portfolio)
    for order in approved_orders:
        fill_price = cost_model.apply(close_price, order, volume)
        portfolio.execute_order(ticker, shares, fill_price, T)
    portfolio.mark_to_market(close_prices[T], T)
```

**Why this design:** Pure vectorized approaches hide lookahead prevention inside `.shift(1)` — hard to audit. Pure event-driven is slow for large universes. The hybrid gives vectorized speed for signal computation and realistic execution mechanics, with a clean separation that mirrors how real quant systems split alpha generation from order management.

### Lookahead Bias Prevention

`LookaheadBarrier` is an API-level enforcement class — not a convention. Every strategy receives a barrier object, not the raw DataFrame. Two access methods:

- `barrier.get_shifted_data()` — returns the full OHLCV frame with `.shift(1)` applied; signal at date T uses data from T-1
- `barrier.get_history(as_of)` — returns all rows strictly before `as_of`; used for model fitting (e.g., cointegration training window)

The engine raises `LookaheadBiasError` if a strategy attempts to access data past the cutoff, and asserts that `signals_df` contains no NaN values before the loop starts.

---

## File Structure

```
backtesting/
├── config.py                  # BacktestConfig dataclass
├── universe.py                # DEV_UNIVERSE (20 tickers), FULL_UNIVERSE
├── run_backtest.py            # CLI entry point
├── core/
│   ├── engine.py              # LookaheadBarrier + BacktestEngine
│   ├── data_loader.py         # yfinance fetch, forward-fill, quality checks
│   ├── cost_model.py          # bid-ask spread, market impact, slippage, commission
│   ├── portfolio.py           # cash, positions, mark-to-market, transaction log
│   ├── risk_manager.py        # position size limits, margin checks
│   └── performance.py         # all metrics computed from scratch
├── strategies/
│   ├── base.py                # abstract Strategy base class
│   ├── momentum.py            # 12-1 month cross-sectional momentum
│   ├── mean_reversion.py      # Bollinger Bands mean reversion
│   ├── pairs_trading.py       # Kalman filter pairs trading
│   └── multi_factor.py        # momentum + low-vol + reversal
├── reporting/
│   └── tearsheet.py           # 5-panel matplotlib tearsheet
└── tests/                     # 54 tests, pytest
```

---

## Market Simulation

The cost model applies four layers to every fill:

| Component | Formula | Always? |
|---|---|---|
| Bid-ask spread | buy: `close × (1 + half_spread)` / sell: `close × (1 − half_spread)` | Yes |
| Commission | `$1 flat + 0.01% of notional` | Yes |
| Market impact | `0.1% × √(order_notional / avg_daily_volume_notional)` | Only if order > 0.5% of 20-day avg volume |
| Slippage | `abs(Normal(0, 0.02%))` applied adversarially | Yes |

Slippage sign convention: `fill_price × (1 + slippage)` for buys, `fill_price × (1 − slippage)` for sells. The absolute value ensures slippage is always a cost.

**Note:** Close prices are used as fill price proxies. Next-day open execution would add approximately 0.1–0.3% additional slippage not captured here.

---

## Strategies

### Momentum (`strategies/momentum.py`)
Cross-sectional momentum: monthly rebalance on the last business day of each month. Ranks all tickers by 12-minus-1 month return (skips the most recent month to avoid short-term mean-reversion). Longs top quintile, shorts bottom quintile, equal weight per leg summing to ±0.5.

Warmup: 252 trading days (1 year of history required). Signals are 0 during warmup.

*Academic basis: Jegadeesh & Titman (1993)*

### Mean Reversion — Bollinger Bands (`strategies/mean_reversion.py`)
Daily rebalancing. Computes 20-day rolling mean and standard deviation per ticker. Enters long when price falls below the lower band (mean − 2σ), short when price rises above the upper band (mean + 2σ). Exits when price crosses back through the rolling mean. Stop-loss at ±3σ (trailing, anchored to the current rolling mean). Weight ±0.1 per position.

A `stopped` flag prevents immediate re-entry on the same bar after a stop fires.

*Academic basis: Bollinger (2001)*

### Pairs Trading — Kalman Filter (`strategies/pairs_trading.py`)
Selects cointegrated pairs using the Engle-Granger test (p < 0.05) on a 2-year training window. Estimates a dynamic hedge ratio β using a custom scalar Kalman filter (pure numpy — no pykalman dependency). The hedge ratio is baked directly into the signal weights so the engine sees only per-ticker target weights.

Spread = `price_A − β × price_B`, z-scored over a 20-day rolling window. Entry at ±2σ, exit at 0σ, stop at ±3σ.

Default pairs: AAPL/MSFT, JPM/GS, XOM/CVX, NVDA/AMD, WMT/HD.

*Academic basis: Engle & Granger (1987)*

### Multi-Factor (`strategies/multi_factor.py`)
Monthly-rebalanced long-short model. Combines three factors, each cross-sectionally z-scored:

| Factor | Definition | Weight |
|---|---|---|
| Momentum | 12-minus-1 month return | 0.5 |
| Low Volatility | Negative 20-day realized vol | 0.3 |
| Short-term Reversal | Negative 1-month return | 0.2 |

Longs top quintile, shorts bottom quintile, ±0.5 total weight per leg.

P/B and ROE were omitted — yfinance fundamental data is unreliable for backtesting. Low volatility (Ang et al. 2006) and short-term reversal (Jegadeesh 1990) are academically validated replacements.

---

## Performance Metrics

All metrics computed from scratch in `core/performance.py`:

| Metric | Definition |
|---|---|
| Annualized return | `(1 + total_return)^(252/n_days) − 1` |
| Annualized volatility | `daily_returns.std() × √252` |
| Sharpe ratio | `(ann_return − 0.05) / ann_vol` |
| Sortino ratio | `(ann_return − 0.05) / downside_vol`; downside_vol = RMS of `min(r, 0)` × √252 |
| Max drawdown | `min(nav / nav.cummax() − 1)` |
| Max drawdown duration | Longest consecutive days below previous high |
| Calmar ratio | `CAGR / abs(max_drawdown)` |
| Hit rate | `(daily_returns > 0).mean()` |
| Monthly returns matrix | Compounded monthly returns, pivoted to year × month |

The Sortino downside deviation uses the RMS of `min(r_t, 0)` across all trading days — the standard Sortino & van der Meer (1991) definition, not the sample standard deviation of negative returns only.

---

## Validation Gate

Before running strategies, a SPY buy-and-hold validation confirms the NAV accounting is correct:

```
=== SPY Buy-and-Hold Validation ===
SPY actual return:    145.0980%
Engine return:        145.1852%
Gap (after costs):    0.0873%
PASS: engine correctly tracks SPY return within 0.5% after costs.
```

The 0.09% gap is attributable to the initial purchase cost (bid-ask spread + commission). If this gap exceeds 0.5%, there is a bug in `Portfolio.execute_order` or `mark_to_market`.

---

## Setup

```bash
pip install -r requirements.txt
```

**Data universe (dev):** 20 liquid large-caps — AAPL, MSFT, GOOGL, AMZN, META, JPM, GS, BAC, XOM, CVX, JNJ, PFE, WMT, HD, TSLA, NVDA, AMD, SPY, QQQ, GLD — over 2018–2024.

---

## Usage

```bash
# Run SPY buy-and-hold validation (correctness check)
python run_backtest.py --validate

# Run a strategy
python run_backtest.py --strategy momentum
python run_backtest.py --strategy bollinger
python run_backtest.py --strategy pairs
python run_backtest.py --strategy multifactor

# Custom date range and capital
python run_backtest.py --strategy momentum --start 2020-01-01 --end 2023-12-31 --capital 500000
```

Output: performance metrics to stdout + tearsheet PNG saved to `output/<strategy>_tearsheet.png`.

```bash
# Run all tests
pytest tests/ -v
```

---

## Known Limitations

**Survivorship bias:** The dev universe uses current large-caps. Companies that failed or were delisted between 2018–2024 are excluded, which inflates strategy returns. Using historical S&P 500 constituents would correct this.

**Fill price proxy:** Close prices approximate fill prices. In live trading, orders execute at the next day's open (or intraday VWAP), adding approximately 0.1–0.3% additional slippage not captured here.

**Fundamental data:** The multi-factor strategy omits P/B ratio and ROE. Yahoo Finance's historical fundamental data has survivorship and point-in-time issues that make it unsuitable for backtesting. Low-vol and reversal factors are used as academically validated alternatives.

**No walk-forward optimization:** Strategy parameters (Bollinger window, momentum lookback) are fixed. Walk-forward validation would be required before drawing conclusions about out-of-sample performance.

**No Monte Carlo:** Equity curve bootstrapping to estimate 5th-percentile outcomes is not implemented.

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Data source | yfinance (free, auto-adjusted) |
| Statistical tests | statsmodels (cointegration) |
| Kalman filter | Custom numpy implementation |
| Visualization | matplotlib, seaborn |
| Testing | pytest |

# Dashboard, Walk-Forward, and Monte Carlo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit live-runner dashboard, a generic walk-forward optimizer for all 4 strategies, and a Monte Carlo simulator with a dedicated dashboard tab.

**Architecture:** Three additive modules (`core/walk_forward.py`, `core/monte_carlo.py`, `app.py`) sit on top of the existing engine. Strategies gain a `params: dict = None` kwarg so the optimizer can pass parameter combos without touching any existing call sites. The dashboard imports directly from `core/` and `strategies/` — `run_backtest.py` is untouched.

**Tech Stack:** Python 3.11+, pandas, numpy, matplotlib, streamlit>=1.35, statsmodels, scipy (all already in requirements.txt except streamlit)

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `strategies/mean_reversion.py` | Modify | Add `param_grid` + `params` kwarg to `BollingerStrategy` |
| `strategies/momentum.py` | Modify | Add `param_grid` + `params` kwarg to `MomentumStrategy` |
| `strategies/pairs_trading.py` | Modify | Add `param_grid` + `params` kwarg to `PairsTradingStrategy` |
| `strategies/multi_factor.py` | Modify | Add `param_grid` + `params` kwarg to `MultiFactorStrategy` |
| `core/walk_forward.py` | Create | `WalkForwardOptimizer` — sliding window param evaluation |
| `core/monte_carlo.py` | Create | `MonteCarloSimulator` — IID bootstrap + percentile bands |
| `app.py` | Create | Streamlit entry point — two tabs, no business logic |
| `tests/test_walk_forward.py` | Create | Tests for `WalkForwardOptimizer` |
| `tests/test_monte_carlo.py` | Create | Tests for `MonteCarloSimulator` |
| `requirements.txt` | Modify | Add `streamlit>=1.35` |

No changes to `core/engine.py`, `core/portfolio.py`, `core/cost_model.py`, `core/performance.py`, `reporting/tearsheet.py`, `run_backtest.py`, or `config.py`.

---

## Task 1: BollingerStrategy param interface

**Files:**
- Modify: `strategies/mean_reversion.py`
- Test: `tests/test_strategies.py` (add to existing file)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategies.py`:

```python
def test_bollinger_accepts_custom_params(synthetic_data, basic_config):
    """params dict overrides default window and entry_z."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = BollingerStrategy(basic_config, params={"window": 10, "entry_z": 1.5})
    assert strategy.window == 10
    assert strategy.entry_z == 1.5
    signals = strategy.generate_signals(barrier)
    assert not signals.isnull().any().any()


def test_bollinger_defaults_unchanged_without_params(synthetic_data, basic_config):
    """No params → window=20, entry_z=2.0 (existing behavior preserved)."""
    strategy = BollingerStrategy(basic_config)
    assert strategy.window == 20
    assert strategy.entry_z == 2.0


def test_bollinger_has_param_grid():
    assert hasattr(BollingerStrategy, "param_grid")
    assert "window" in BollingerStrategy.param_grid
    assert "entry_z" in BollingerStrategy.param_grid
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_strategies.py::test_bollinger_accepts_custom_params tests/test_strategies.py::test_bollinger_defaults_unchanged_without_params tests/test_strategies.py::test_bollinger_has_param_grid -v
```

Expected: FAIL — `BollingerStrategy.__init__` doesn't accept `params`, no `param_grid` attribute.

- [ ] **Step 3: Implement**

Replace `strategies/mean_reversion.py` with:

```python
import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class BollingerStrategy(Strategy):
    """
    Mean-reversion using Bollinger Bands.
    Long when price < lower band, short when price > upper band.
    Exit when price crosses back through the mean.
    Stop-loss at rolling mean ±3σ (trailing stop).

    Academic basis: Bollinger (2001).
    """

    param_grid = {
        "window": [10, 15, 20, 25, 30],
        "entry_z": [1.5, 2.0, 2.5, 3.0],
    }

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        self.window = _p.get("window", 20)
        self.entry_z = _p.get("entry_z", 2.0)

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        upper = rolling_mean + self.entry_z * rolling_std
        lower = rolling_mean - self.entry_z * rolling_std
        stop_upper = rolling_mean + 3.0 * rolling_std
        stop_lower = rolling_mean - 3.0 * rolling_std

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        current_signal = {t: 0.0 for t in close.columns}

        for i, date in enumerate(close.index):
            if i < self.window:
                continue
            for ticker in close.columns:
                price = close.loc[date, ticker]
                if pd.isna(price):
                    continue

                mean = rolling_mean.loc[date, ticker]
                lb = lower.loc[date, ticker]
                ub = upper.loc[date, ticker]
                sl_lo = stop_lower.loc[date, ticker]
                sl_hi = stop_upper.loc[date, ticker]

                if pd.isna(mean):
                    continue

                sig = current_signal[ticker]

                stopped = False
                if sig > 0 and price < sl_lo:
                    sig = 0.0
                    stopped = True
                elif sig < 0 and price > sl_hi:
                    sig = 0.0
                    stopped = True

                if sig > 0 and price >= mean:
                    sig = 0.0
                elif sig < 0 and price <= mean:
                    sig = 0.0

                if not stopped and sig == 0.0:
                    if price < lb:
                        sig = 0.1
                    elif price > ub:
                        sig = -0.1

                current_signal[ticker] = sig
                signals.loc[date, ticker] = sig

        return signals.fillna(0.0)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_strategies.py -v -k "bollinger"
```

Expected: all 7 Bollinger tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strategies/mean_reversion.py tests/test_strategies.py
git commit -m "feat: add param_grid and params interface to BollingerStrategy"
```

---

## Task 2: MomentumStrategy param interface

**Files:**
- Modify: `strategies/momentum.py`
- Test: `tests/test_strategies.py` (add to existing file)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategies.py`:

```python
def test_momentum_accepts_custom_params(synthetic_data, basic_config):
    """params={'lookback_months': 6} → warmup is 126 days (6*21), not 252."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MomentumStrategy(basic_config, params={"lookback_months": 6, "skip_months": 1})
    assert strategy.lookback_days == 126
    assert strategy.skip_days == 21
    signals = strategy.generate_signals(barrier)
    assert (signals.iloc[:126] == 0.0).all().all()
    assert not signals.isnull().any().any()


def test_momentum_defaults_unchanged_without_params(basic_config):
    """No params → lookback_days=252, skip_days=21."""
    strategy = MomentumStrategy(basic_config)
    assert strategy.lookback_days == 252
    assert strategy.skip_days == 21


def test_momentum_has_param_grid():
    assert hasattr(MomentumStrategy, "param_grid")
    assert "lookback_months" in MomentumStrategy.param_grid
    assert "skip_months" in MomentumStrategy.param_grid
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_strategies.py::test_momentum_accepts_custom_params tests/test_strategies.py::test_momentum_defaults_unchanged_without_params tests/test_strategies.py::test_momentum_has_param_grid -v
```

Expected: FAIL — no `params` kwarg, no `param_grid`, no `lookback_days` attribute.

- [ ] **Step 3: Implement**

Replace `strategies/momentum.py` with:

```python
import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum: monthly rebalance, long top quintile,
    short bottom quintile ranked by lookback-minus-skip month return.

    Academic basis: Jegadeesh & Titman (1993).
    """

    param_grid = {
        "lookback_months": [6, 9, 12, 15],
        "skip_months": [1, 2],
    }

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        self.lookback_days = _p.get("lookback_months", 12) * 21
        self.skip_days = _p.get("skip_months", 1) * 21

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        ret_lookback = close.pct_change(self.lookback_days)
        ret_skip = close.pct_change(self.skip_days)
        momentum = ret_lookback - ret_skip

        month_ends = close.resample("BME").last().index

        for date in month_ends:
            if date not in close.index:
                continue
            scores = momentum.loc[date].dropna()
            if len(scores) < 5:
                continue

            n = len(scores)
            quintile_size = max(1, n // 5)

            ranked = scores.rank(ascending=True)
            long_tickers = ranked.nlargest(quintile_size).index
            short_tickers = ranked.nsmallest(quintile_size).index

            long_weight = 0.5 / len(long_tickers)
            short_weight = -0.5 / len(short_tickers)

            signals.loc[date, :] = 0.0
            for ticker in long_tickers:
                if ticker in signals.columns:
                    signals.loc[date, ticker] = long_weight
            for ticker in short_tickers:
                if ticker in signals.columns:
                    signals.loc[date, ticker] = short_weight

        signals = signals.ffill().fillna(0.0)
        signals.iloc[:self.lookback_days] = 0.0

        return signals
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_strategies.py -v -k "momentum"
```

Expected: all Momentum tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strategies/momentum.py tests/test_strategies.py
git commit -m "feat: add param_grid and params interface to MomentumStrategy"
```

---

## Task 3: PairsTradingStrategy param interface

**Files:**
- Modify: `strategies/pairs_trading.py`
- Test: `tests/test_strategies.py` (add to existing file)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategies.py`:

```python
def test_pairs_accepts_custom_params(basic_config):
    """params dict overrides z_entry and z_exit."""
    strategy = PairsTradingStrategy(basic_config, params={"entry_z": 2.5, "exit_z": 0.5})
    assert strategy.z_entry == 2.5
    assert strategy.z_exit == 0.5


def test_pairs_defaults_unchanged_without_params(basic_config):
    """No params → z_entry=2.0, z_exit=0.0."""
    strategy = PairsTradingStrategy(basic_config)
    assert strategy.z_entry == 2.0
    assert strategy.z_exit == 0.0


def test_pairs_has_param_grid():
    assert hasattr(PairsTradingStrategy, "param_grid")
    assert "entry_z" in PairsTradingStrategy.param_grid
    assert "exit_z" in PairsTradingStrategy.param_grid
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_strategies.py::test_pairs_accepts_custom_params tests/test_strategies.py::test_pairs_defaults_unchanged_without_params tests/test_strategies.py::test_pairs_has_param_grid -v
```

Expected: FAIL — no `params` kwarg, no `param_grid`.

- [ ] **Step 3: Implement**

In `strategies/pairs_trading.py`, update `PairsTradingStrategy` class only (keep `_kalman_hedge_ratio` function unchanged):

```python
class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage via cointegrated pairs.
    Uses a scalar Kalman filter to estimate the dynamic hedge ratio β.
    Spread = price_A - β * price_B, traded when z-score exceeds ±entry_z.

    Academic basis: Engle & Granger (1987).
    """

    DEFAULT_PAIRS = [
        ("AAPL", "MSFT"), ("JPM", "GS"), ("XOM", "CVX"),
        ("NVDA", "AMD"), ("WMT", "HD"),
    ]

    param_grid = {
        "entry_z": [1.5, 2.0, 2.5, 3.0],
        "exit_z": [0.0, 0.5],
    }

    def __init__(
        self,
        config: BacktestConfig,
        params: dict = None,
        pairs: List[Tuple[str, str]] = None,
        training_days: int = 504,
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        z_stop: float = 3.0,
        spread_window: int = 20,
    ):
        self.config = config
        self.pairs = pairs if pairs is not None else self.DEFAULT_PAIRS
        _p = params or {}
        self.training_days = _p.get("training_days", training_days)
        self.z_entry = _p.get("entry_z", z_entry)
        self.z_exit = _p.get("exit_z", z_exit)
        self.z_stop = z_stop
        self.spread_window = spread_window
```

Leave `generate_signals` unchanged — it already uses `self.z_entry`, `self.z_exit`, `self.z_stop`, `self.training_days`.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_strategies.py -v -k "pairs"
```

Expected: all Pairs tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strategies/pairs_trading.py tests/test_strategies.py
git commit -m "feat: add param_grid and params interface to PairsTradingStrategy"
```

---

## Task 4: MultiFactorStrategy param interface

**Files:**
- Modify: `strategies/multi_factor.py`
- Test: `tests/test_strategies.py` (add to existing file)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategies.py`:

```python
def test_multifactor_accepts_custom_params(synthetic_data, basic_config):
    """params dict overrides factor weights; reversal_wt = 1 - momentum_wt - lowvol_wt."""
    barrier = LookaheadBarrier(synthetic_data)
    strategy = MultiFactorStrategy(basic_config, params={"momentum_wt": 0.33, "lowvol_wt": 0.33})
    assert abs(strategy.factor_weights["momentum"] - 0.33) < 1e-9
    assert abs(strategy.factor_weights["low_vol"] - 0.33) < 1e-9
    assert abs(strategy.factor_weights["reversal"] - 0.34) < 1e-9
    signals = strategy.generate_signals(barrier)
    assert not signals.isnull().any().any()


def test_multifactor_defaults_unchanged_without_params(basic_config):
    """No params → momentum=0.5, low_vol=0.3, reversal=0.2."""
    strategy = MultiFactorStrategy(basic_config)
    assert strategy.factor_weights == {"momentum": 0.5, "low_vol": 0.3, "reversal": 0.2}


def test_multifactor_has_param_grid():
    assert hasattr(MultiFactorStrategy, "param_grid")
    assert "momentum_wt" in MultiFactorStrategy.param_grid
    assert "lowvol_wt" in MultiFactorStrategy.param_grid
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_strategies.py::test_multifactor_accepts_custom_params tests/test_strategies.py::test_multifactor_defaults_unchanged_without_params tests/test_strategies.py::test_multifactor_has_param_grid -v
```

Expected: FAIL — no `params` kwarg, no `param_grid`, no `factor_weights` attribute.

- [ ] **Step 3: Implement**

Replace `strategies/multi_factor.py` with:

```python
import pandas as pd
import numpy as np
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row cross-sectionally (subtract mean, divide by std)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


class MultiFactorStrategy(Strategy):
    """
    Monthly-rebalanced long-short factor model.
    Factors: momentum (12-1M return), low volatility (neg 20-day vol),
    short-term reversal (neg 1M return). All cross-sectionally z-scored.

    Note: P/B and ROE omitted — yfinance fundamental data is unreliable.
    Low-vol (Ang et al. 2006) and reversal (Jegadeesh 1990) substituted.
    """

    param_grid = {
        "momentum_wt": [0.33, 0.5, 0.7],
        "lowvol_wt": [0.1, 0.3, 0.5],
    }

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        _p = params or {}
        momentum_wt = _p.get("momentum_wt", 0.5)
        lowvol_wt = _p.get("lowvol_wt", 0.3)
        reversal_wt = round(1.0 - momentum_wt - lowvol_wt, 10)
        self.factor_weights = {
            "momentum": momentum_wt,
            "low_vol": lowvol_wt,
            "reversal": reversal_wt,
        }

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].ffill()

        ret_12m = close.pct_change(252)
        ret_1m = close.pct_change(21)
        momentum_raw = ret_12m - ret_1m
        low_vol_raw = -close.pct_change().rolling(20).std()
        reversal_raw = -ret_1m

        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        month_ends = close.resample("BME").last().index

        for date in month_ends:
            if date not in close.index:
                continue

            factors = pd.DataFrame({
                "momentum": momentum_raw.loc[date],
                "low_vol": low_vol_raw.loc[date],
                "reversal": reversal_raw.loc[date],
            }).dropna()

            if len(factors) < 5:
                continue

            zscored = _cross_sectional_zscore(factors)

            combined = (
                zscored["momentum"] * self.factor_weights["momentum"]
                + zscored["low_vol"] * self.factor_weights["low_vol"]
                + zscored["reversal"] * self.factor_weights["reversal"]
            )

            n = len(combined)
            quintile_size = max(1, n // 5)
            long_tickers = combined.nlargest(quintile_size).index
            short_tickers = combined.nsmallest(quintile_size).index

            long_weight = 0.5 / len(long_tickers)
            short_weight = -0.5 / len(short_tickers)

            signals.loc[date, :] = 0.0
            for t in long_tickers:
                if t in signals.columns:
                    signals.loc[date, t] = long_weight
            for t in short_tickers:
                if t in signals.columns:
                    signals.loc[date, t] = short_weight

        signals = signals.ffill().fillna(0.0)
        signals.iloc[:252] = 0.0

        return signals
```

- [ ] **Step 4: Run all strategy tests to confirm nothing regressed**

```bash
pytest tests/test_strategies.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strategies/multi_factor.py tests/test_strategies.py
git commit -m "feat: add param_grid and params interface to MultiFactorStrategy"
```

---

## Task 5: WalkForwardOptimizer

**Files:**
- Create: `core/walk_forward.py`
- Create: `tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_walk_forward.py`:

```python
import itertools
import pandas as pd
import numpy as np
import pytest
from config import BacktestConfig
from core.engine import LookaheadBarrier
from strategies.base import Strategy


class TinyStrategy(Strategy):
    """Minimal strategy for fast WFO tests — no computation, instant signals."""
    param_grid = {"weight": [0.05, 0.1]}

    def __init__(self, config: BacktestConfig, params: dict = None):
        self.config = config
        self.weight = (params or {}).get("weight", 0.05)

    def generate_signals(self, barrier: LookaheadBarrier) -> pd.DataFrame:
        data = barrier.get_shifted_data()
        close = data["Close"].fillna(0.0)
        return pd.DataFrame(self.weight, index=close.index, columns=close.columns)


def test_wfo_returns_dataframe(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert isinstance(results, pd.DataFrame)


def test_wfo_result_has_required_columns(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert "weight" in results.columns
    assert "in_sample_sharpe" in results.columns
    assert "out_sample_sharpe" in results.columns
    assert "degradation" in results.columns


def test_wfo_one_row_per_combo(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    assert len(results) == 2  # TinyStrategy has 2 combos: weight=[0.05, 0.1]


def test_wfo_sorted_by_oos_sharpe(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    oos = results["out_sample_sharpe"].values
    assert list(oos) == sorted(oos, reverse=True)


def test_wfo_degradation_equals_diff(synthetic_data, basic_config):
    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(TinyStrategy, synthetic_data, basic_config)
    results = optimizer.run(train_years=1, test_months=3)
    for _, row in results.iterrows():
        expected = row["in_sample_sharpe"] - row["out_sample_sharpe"]
        assert abs(row["degradation"] - expected) < 1e-9


def test_wfo_filters_invalid_multifactor_combos():
    from core.walk_forward import WalkForwardOptimizer
    combos = [
        {"momentum_wt": 0.7, "lowvol_wt": 0.5},  # 1.2 → invalid
        {"momentum_wt": 0.7, "lowvol_wt": 0.3},  # 1.0 → valid
        {"momentum_wt": 0.5, "lowvol_wt": 0.3},  # 0.8 → valid
    ]
    filtered = WalkForwardOptimizer._filter_combos(combos)
    assert len(filtered) == 2
    assert not any(
        c["momentum_wt"] + c["lowvol_wt"] > 1.0
        for c in filtered
    )


def test_wfo_base_params_merged_into_combo(synthetic_data, basic_config):
    """base_params are passed to every strategy instantiation alongside grid combo."""
    received = []

    class RecordingStrategy(Strategy):
        param_grid = {"weight": [0.05]}

        def __init__(self, config, params=None):
            self.config = config
            received.append(dict(params or {}))

        def generate_signals(self, barrier):
            data = barrier.get_shifted_data()
            close = data["Close"].fillna(0.0)
            return pd.DataFrame(0.05, index=close.index, columns=close.columns)

    from core.walk_forward import WalkForwardOptimizer
    optimizer = WalkForwardOptimizer(
        RecordingStrategy, synthetic_data, basic_config,
        base_params={"fixed_key": "fixed_val"}
    )
    optimizer.run(train_years=1, test_months=3)
    assert all("fixed_key" in p for p in received)
    assert all(p["fixed_key"] == "fixed_val" for p in received)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_walk_forward.py -v
```

Expected: FAIL — `core/walk_forward.py` doesn't exist.

- [ ] **Step 3: Implement**

Create `core/walk_forward.py`:

```python
import itertools
import numpy as np
import pandas as pd
from config import BacktestConfig
from core.engine import BacktestEngine
from core.performance import PerformanceAnalyzer


class WalkForwardOptimizer:
    """
    Evaluates a strategy's param_grid across sliding train/test windows.
    Returns one row per param combo with averaged in-sample and OOS Sharpe.

    train_years=2, test_months=6 → ~2-year train window, 6-month OOS test,
    sliding forward by test_months each step.
    """

    def __init__(
        self,
        strategy_cls,
        data: pd.DataFrame,
        base_config: BacktestConfig,
        base_params: dict = None,
    ):
        self.strategy_cls = strategy_cls
        self.data = data
        self.base_config = base_config
        self.base_params = base_params or {}

    def run(self, train_years: int = 2, test_months: int = 6) -> pd.DataFrame:
        grid = self.strategy_cls.param_grid
        param_names = list(grid.keys())
        all_combos = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*grid.values())
        ]
        all_combos = self._filter_combos(all_combos)
        windows = self._get_windows(train_years, test_months)

        results = []
        for combo in all_combos:
            merged = {**self.base_params, **combo}
            in_sharpes, oos_sharpes = [], []

            for train_sl, test_sl in windows:
                for sl, sharpe_list in [(train_sl, in_sharpes), (test_sl, oos_sharpes)]:
                    slice_data = self.data.iloc[sl.start:sl.stop]
                    strategy = self.strategy_cls(self.base_config, params=merged)
                    engine = BacktestEngine(self.base_config, slice_data)
                    nav = engine.run(strategy)
                    metrics = PerformanceAnalyzer(
                        nav, self.base_config.risk_free_rate
                    ).compute()
                    sharpe_list.append(metrics["sharpe_ratio"])

            results.append({
                **combo,
                "in_sample_sharpe": float(np.mean(in_sharpes)),
                "out_sample_sharpe": float(np.mean(oos_sharpes)),
                "degradation": float(np.mean(in_sharpes) - np.mean(oos_sharpes)),
            })

        return (
            pd.DataFrame(results)
            .sort_values("out_sample_sharpe", ascending=False)
            .reset_index(drop=True)
        )

    def _get_windows(self, train_years: int, test_months: int):
        """Returns list of (train_slice, test_slice) as range objects."""
        train_size = train_years * 252
        test_size = test_months * 21
        total = len(self.data)

        windows = []
        start = 0
        while start + train_size + test_size <= total:
            windows.append((
                range(start, start + train_size),
                range(start + train_size, start + train_size + test_size),
            ))
            start += test_size
        return windows

    @staticmethod
    def _filter_combos(combos: list) -> list:
        """Remove MultiFactorStrategy combos where momentum_wt + lowvol_wt > 1.0."""
        result = []
        for c in combos:
            if "momentum_wt" in c and "lowvol_wt" in c:
                if c["momentum_wt"] + c["lowvol_wt"] > 1.0:
                    continue
            result.append(c)
        return result
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_walk_forward.py -v
```

Expected: all 7 WFO tests PASS.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add core/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: WalkForwardOptimizer with sliding windows and combo filtering"
```

---

## Task 6: MonteCarloSimulator

**Files:**
- Create: `core/monte_carlo.py`
- Create: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monte_carlo.py`:

```python
import numpy as np
import pandas as pd
import pytest


def _make_nav(n=100, start=1_000_000, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    values = start * np.cumprod(1 + rets)
    return pd.Series(values, index=pd.bdate_range("2020-01-02", periods=n))


def test_mc_run_returns_dataframe():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc = MonteCarloSimulator(nav, n_simulations=50, seed=42)
    sims = mc.run()
    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (100, 50)


def test_mc_anchored_at_initial_nav():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100, start=500_000)
    mc = MonteCarloSimulator(nav, n_simulations=20, seed=0)
    sims = mc.run()
    assert (sims.iloc[0] == nav.iloc[0]).all()


def test_mc_index_matches_nav():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc = MonteCarloSimulator(nav, n_simulations=10, seed=1)
    sims = mc.run()
    assert sims.index.equals(nav.index)


def test_mc_percentile_bands_columns():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc = MonteCarloSimulator(nav, n_simulations=50, seed=42)
    bands = mc.percentile_bands()
    assert set(["p5", "p50", "p95", "actual"]).issubset(bands.columns)


def test_mc_p5_leq_p50_leq_p95():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(200)
    mc = MonteCarloSimulator(nav, n_simulations=200, seed=42)
    bands = mc.percentile_bands()
    assert (bands["p5"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p95"]).all()


def test_mc_summary_metrics_keys():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc = MonteCarloSimulator(nav, n_simulations=50, seed=42)
    metrics = mc.summary_metrics()
    assert "median_terminal_nav" in metrics
    assert "p5_terminal_nav" in metrics
    assert "p5_max_drawdown" in metrics
    assert "prob_of_loss" in metrics


def test_mc_p5_terminal_nav_lt_median():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(200)
    mc = MonteCarloSimulator(nav, n_simulations=500, seed=42)
    metrics = mc.summary_metrics()
    assert metrics["p5_terminal_nav"] <= metrics["median_terminal_nav"]


def test_mc_seeded_reproducible():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc1 = MonteCarloSimulator(nav, n_simulations=50, seed=99)
    mc2 = MonteCarloSimulator(nav, n_simulations=50, seed=99)
    pd.testing.assert_frame_equal(mc1.run(), mc2.run())


def test_mc_p5_max_drawdown_is_negative():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(200)
    mc = MonteCarloSimulator(nav, n_simulations=200, seed=42)
    metrics = mc.summary_metrics()
    assert metrics["p5_max_drawdown"] <= 0.0


def test_mc_prob_of_loss_between_zero_and_one():
    from core.monte_carlo import MonteCarloSimulator
    nav = _make_nav(100)
    mc = MonteCarloSimulator(nav, n_simulations=100, seed=42)
    metrics = mc.summary_metrics()
    assert 0.0 <= metrics["prob_of_loss"] <= 1.0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: FAIL — `core/monte_carlo.py` doesn't exist.

- [ ] **Step 3: Implement**

Create `core/monte_carlo.py`:

```python
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """
    IID bootstrap of daily returns to generate simulated equity curves.
    Resamples with replacement from the historical daily return series.
    """

    def __init__(self, nav: pd.Series, n_simulations: int = 1000, seed: int = 42):
        self.nav = nav
        self.n_simulations = n_simulations
        self.seed = seed
        self._simulations: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """
        Returns DataFrame of shape (len(nav), n_simulations).
        Each column is one simulated NAV curve anchored at nav.iloc[0].
        Calling run() again with the same seed returns identical results.
        """
        rng = np.random.default_rng(self.seed)
        returns = self.nav.pct_change().dropna().values
        n = len(returns)

        # Sample with replacement: shape (n, n_simulations)
        sampled = rng.choice(returns, size=(n, self.n_simulations), replace=True)

        # Compound returns into NAV curves
        curves = np.cumprod(1 + sampled, axis=0) * self.nav.iloc[0]

        # Prepend the known starting NAV so index aligns with self.nav
        start_row = np.full((1, self.n_simulations), self.nav.iloc[0])
        curves = np.vstack([start_row, curves])

        self._simulations = pd.DataFrame(
            curves,
            index=self.nav.index,
            columns=range(self.n_simulations),
        )
        return self._simulations

    def percentile_bands(self, percentiles: list = None) -> pd.DataFrame:
        """
        Returns DataFrame with columns p5, p50, p95 and actual NAV.
        Calls run() automatically if not already called.
        """
        if percentiles is None:
            percentiles = [5, 50, 95]
        if self._simulations is None:
            self.run()
        result = pd.DataFrame(index=self.nav.index)
        for p in percentiles:
            result[f"p{p}"] = np.percentile(self._simulations.values, p, axis=1)
        result["actual"] = self.nav.values
        return result

    def summary_metrics(self) -> dict:
        """
        Returns:
          median_terminal_nav  — p50 of terminal NAV across simulations
          p5_terminal_nav      — p5 of terminal NAV
          p5_max_drawdown      — p5 of per-simulation max drawdown (≤ 0)
          prob_of_loss         — fraction of simulations ending below starting NAV
        """
        if self._simulations is None:
            self.run()

        terminal_navs = self._simulations.iloc[-1].values

        # Per-simulation max drawdown
        sims_arr = self._simulations.values
        running_max = np.maximum.accumulate(sims_arr, axis=0)
        drawdowns = sims_arr / running_max - 1
        max_drawdowns = drawdowns.min(axis=0)

        return {
            "median_terminal_nav": float(np.percentile(terminal_navs, 50)),
            "p5_terminal_nav": float(np.percentile(terminal_navs, 5)),
            "p5_max_drawdown": float(np.percentile(max_drawdowns, 5)),
            "prob_of_loss": float((terminal_navs < self.nav.iloc[0]).mean()),
        }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: all 10 Monte Carlo tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add core/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat: MonteCarloSimulator with IID bootstrap and percentile bands"
```

---

## Task 7: Streamlit dashboard

**Files:**
- Create: `app.py`

No automated tests — manual verification via browser.

- [ ] **Step 1: Create `app.py`**

```python
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from config import BacktestConfig
from universe import DEV_UNIVERSE
from core.data_loader import DataLoader
from core.engine import BacktestEngine
from core.performance import PerformanceAnalyzer
from core.walk_forward import WalkForwardOptimizer
from core.monte_carlo import MonteCarloSimulator
from reporting.tearsheet import TearsheetGenerator
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import BollingerStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.multi_factor import MultiFactorStrategy

STRATEGIES = {
    "momentum": MomentumStrategy,
    "bollinger": BollingerStrategy,
    "pairs": PairsTradingStrategy,
    "multifactor": MultiFactorStrategy,
}

# PairsTradingStrategy needs training_days reduced for WFO slices
# (default 504 days would consume the entire train window)
WFO_BASE_PARAMS = {
    "pairs": {"training_days": 252},
}

st.set_page_config(page_title="Backtesting Framework", layout="wide")
st.title("Backtesting Framework")

tab1, tab2 = st.tabs(["Strategy Results", "Monte Carlo"])

with tab1:
    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 2, 1])
    with col1:
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
    with col2:
        start_date = st.date_input("Start", value=pd.Timestamp("2018-01-01"))
    with col3:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))
    with col4:
        capital = st.number_input("Initial Capital ($)", value=1_000_000, step=100_000, min_value=10_000)
    with col5:
        st.write("")
        st.write("")
        run_clicked = st.button("▶ Run Backtest", use_container_width=True)

    if run_clicked:
        cfg = BacktestConfig(
            tickers=DEV_UNIVERSE,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=float(capital),
        )
        with st.spinner("Fetching data and running backtest…"):
            loader = DataLoader(cfg)
            data = loader.fetch()
            strategy = STRATEGIES[strategy_name](cfg)
            engine = BacktestEngine(cfg, data)
            nav = engine.run(strategy)
            spy_nav = data["Close"]["SPY"].reindex(nav.index).ffill()
            pa = PerformanceAnalyzer(nav, risk_free_rate=cfg.risk_free_rate)
            metrics = pa.compute()

            os.makedirs("output", exist_ok=True)
            output_path = f"output/{strategy_name}_tearsheet.png"
            gen = TearsheetGenerator(strategy_name=strategy_name.title())
            gen.render(metrics, nav, spy_nav, output_path)

            st.session_state["nav"] = nav
            st.session_state["metrics"] = metrics
            st.session_state["spy_nav"] = spy_nav
            st.session_state["strategy_name"] = strategy_name
            st.session_state["data"] = data
            st.session_state["cfg"] = cfg

    if "metrics" in st.session_state:
        m = st.session_state["metrics"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Ann. Return", f"{m['annualized_return']:.1%}")
        c2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
        c3.metric("Sortino", f"{m['sortino_ratio']:.2f}")
        c4.metric("Max Drawdown", f"{m['max_drawdown']:.1%}")
        c5.metric("Calmar", f"{m['calmar_ratio']:.2f}")
        c6.metric("Hit Rate", f"{m['hit_rate']:.1%}")

        sn = st.session_state["strategy_name"]
        img_path = f"output/{sn}_tearsheet.png"
        if os.path.exists(img_path):
            st.image(img_path)

        with st.expander("Walk-Forward Results"):
            if st.button("▶ Run Walk-Forward Analysis"):
                wfo_key = f"wfo_{sn}"
                base_params = WFO_BASE_PARAMS.get(sn, {})
                with st.spinner("Running walk-forward optimization (this takes ~2 min)…"):
                    optimizer = WalkForwardOptimizer(
                        STRATEGIES[sn],
                        st.session_state["data"],
                        st.session_state["cfg"],
                        base_params=base_params,
                    )
                    st.session_state[wfo_key] = optimizer.run()
            wfo_key = f"wfo_{sn}"
            if wfo_key in st.session_state:
                st.dataframe(st.session_state[wfo_key], use_container_width=True)

with tab2:
    if "nav" not in st.session_state:
        st.info("Run a backtest on the Strategy Results tab first to enable Monte Carlo.")
    else:
        n_sims = st.slider("Simulations", min_value=100, max_value=5000, value=1000, step=100)
        with st.spinner("Running Monte Carlo simulation…"):
            mc = MonteCarloSimulator(st.session_state["nav"], n_simulations=n_sims)
            mc.run()
            bands = mc.percentile_bands()
            mc_metrics = mc.summary_metrics()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(
            bands.index, bands["p5"], bands["p95"],
            alpha=0.2, color="steelblue", label="p5–p95 band"
        )
        ax.fill_between(
            bands.index, bands["p5"], bands["p50"],
            alpha=0.1, color="steelblue"
        )
        ax.plot(bands.index, bands["p50"], color="steelblue", linewidth=1.0, label="p50 (median)")
        ax.plot(bands.index, bands["actual"], color="crimson", linewidth=1.5, label="Actual NAV")
        ax.set_title(f"Monte Carlo Simulation ({n_sims:,} paths) — IID Bootstrap")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Median Terminal NAV", f"${mc_metrics['median_terminal_nav']:,.0f}")
        mc2.metric("p5 Terminal NAV", f"${mc_metrics['p5_terminal_nav']:,.0f}")
        mc3.metric("p5 Max Drawdown", f"{mc_metrics['p5_max_drawdown']:.1%}")
        mc4.metric("Probability of Loss", f"{mc_metrics['prob_of_loss']:.1%}")
```

- [ ] **Step 2: Start the dashboard and verify Tab 1**

```bash
streamlit run app.py
```

Open http://localhost:8501. Select "bollinger", leave dates as default, click **▶ Run Backtest**. Verify:
- Spinner appears and disappears
- 6 metric cards show non-zero values
- Tearsheet image renders
- `output/bollinger_tearsheet.png` exists on disk

- [ ] **Step 3: Verify walk-forward**

Open the "Walk-Forward Results" expander, click **▶ Run Walk-Forward Analysis**. Verify:
- Spinner appears (takes ~1-2 min for Bollinger on dev universe)
- Table appears with columns: `window`, `entry_z`, `in_sample_sharpe`, `out_sample_sharpe`, `degradation`
- Table sorted by `out_sample_sharpe` descending

- [ ] **Step 4: Verify Monte Carlo tab**

Click the **Monte Carlo** tab. Verify:
- Band chart renders with p5/p50/p95 envelopes and actual NAV overlay
- 4 metric cards show plausible values
- Move the slider from 1000 to 200 — chart updates and bands widen visibly

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit live-runner dashboard with walk-forward expander and Monte Carlo tab"
```

---

## Task 8: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add streamlit**

Edit `requirements.txt` to add:

```
streamlit>=1.35
```

- [ ] **Step 2: Verify install (optional — skip if streamlit is already installed)**

```bash
pip install -r requirements.txt
```

Expected: no errors.

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests PASS (streamlit addition doesn't affect tests).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add streamlit to requirements"
```

# Extensions Design Spec: Dashboard, Walk-Forward, Monte Carlo
**Date:** 2026-05-16
**Status:** Approved

---

## Overview

Three extensions to the existing backtesting framework:

1. **Streamlit dashboard** (`app.py`) — Live Runner: strategy selector + date range + capital → runs backtest on demand, renders metrics strip + matplotlib tearsheet inline + walk-forward results in expander. Two tabs: Strategy Results and Monte Carlo.
2. **Walk-forward optimizer** (`core/walk_forward.py`) — generic `WalkForwardOptimizer` that evaluates all 4 strategies across sliding train/test windows, surfacing in-sample vs out-of-sample Sharpe degradation.
3. **Monte Carlo simulator** (`core/monte_carlo.py`) — IID bootstrap of daily returns, 1,000 simulations by default, percentile band chart + 4-metric summary.

The existing CLI (`run_backtest.py`), engine, strategies, and tearsheet are unchanged. All new code is additive.

---

## New Files

```
app.py                     # Streamlit entry point — UI only, no business logic
core/
  walk_forward.py          # WalkForwardOptimizer class
  monte_carlo.py           # MonteCarloSimulator class
```

### Modified Files

```
strategies/momentum.py       # add param_grid + accept params kwarg
strategies/mean_reversion.py # add param_grid + accept params kwarg
strategies/pairs_trading.py  # add param_grid + accept params kwarg
strategies/multi_factor.py   # add param_grid + accept params kwarg
requirements.txt             # add streamlit
```

No changes to `core/engine.py`, `core/portfolio.py`, `core/cost_model.py`, `core/performance.py`, `reporting/tearsheet.py`, `run_backtest.py`, or `config.py`.

---

## Walk-Forward Optimizer (`core/walk_forward.py`)

### Interface

```python
class WalkForwardOptimizer:
    def __init__(
        self,
        strategy_cls,            # any Strategy subclass with a param_grid attribute
        data: pd.DataFrame,      # full OHLCV MultiIndex DataFrame
        base_config: BacktestConfig,
    ): ...

    def run(
        self,
        train_years: int = 2,
        test_months: int = 6,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame sorted by out_sample_sharpe descending.
        Columns: all param names + in_sample_sharpe + out_sample_sharpe + degradation
        """
```

### Algorithm

1. Read `strategy_cls.param_grid` — a `dict[str, list]` class attribute.
2. Build all param combos via `itertools.product`. For `MultiFactorStrategy`, filter out any combo where `momentum_wt + lowvol_wt > 1.0` before running.
3. Compute sliding windows from `data.index`: train window = `train_years` years, test window = `test_months` months, slide forward by `test_months` each step. On a 2018–2024 dataset this yields ~6 non-overlapping windows per strategy.
4. For each (combo, window) pair:
   - Slice `data` to the train range, construct `BacktestEngine`, run the strategy, compute Sharpe via `PerformanceAnalyzer`.
   - Repeat for the test range.
5. Average in-sample and OOS Sharpe across windows. Record `degradation = in_sample_sharpe − out_sample_sharpe`.
6. Return one row per param combo, sorted by `out_sample_sharpe` descending.

**Parallelization hook:** Each param combo is fully independent. A `ProcessPoolExecutor` can be dropped in at step 4 if the full run (~540 engine calls across all 4 strategies) is too slow. The sequential version ships first.

### Strategy param grids

Each strategy gains:
- A `param_grid: dict[str, list]` class attribute.
- `params: dict = None` in `__init__`. When `None`, all params fall back to existing defaults — existing CLI behavior is unchanged.
- Internal usage: `self.params.get('window', 20)` pattern throughout.

| Strategy | `param_grid` |
|---|---|
| `BollingerStrategy` | `{"window": [10, 15, 20, 25, 30], "entry_z": [1.5, 2.0, 2.5, 3.0]}` |
| `MomentumStrategy` | `{"lookback_months": [6, 9, 12, 15], "skip_months": [1, 2]}` |
| `PairsTradingStrategy` | `{"entry_z": [1.5, 2.0, 2.5, 3.0], "exit_z": [0.0, 0.5]}` |
| `MultiFactorStrategy` | `{"momentum_wt": [0.33, 0.5, 0.7], "lowvol_wt": [0.1, 0.3, 0.5]}` — invalid combos where sum > 1 filtered out; `reversal_wt = 1 − momentum_wt − lowvol_wt` |

**Combo counts (before filtering):** Bollinger 20, Momentum 8, Pairs 8, MultiFactorStrategy 9 (minus invalid). Total ~540 engine runs across all strategies and windows.

---

## Monte Carlo Simulator (`core/monte_carlo.py`)

### Interface

```python
class MonteCarloSimulator:
    def __init__(
        self,
        nav: pd.Series,
        n_simulations: int = 1000,
        seed: int = 42,
    ): ...

    def run(self) -> pd.DataFrame:
        """
        Returns DataFrame of shape (len(nav), n_simulations).
        Each column is one simulated NAV curve anchored at nav.iloc[0].
        """

    def percentile_bands(
        self,
        percentiles: list[int] = [5, 50, 95],
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns p5, p50, p95 plus the actual NAV series.
        """

    def summary_metrics(self) -> dict:
        """
        Returns:
          median_terminal_nav   float
          p5_terminal_nav       float
          p5_max_drawdown       float   # worst-case drawdown at 5th percentile
          prob_of_loss          float   # fraction of simulations ending below nav.iloc[0]
        """
```

### Bootstrap method

IID resampling: draw `len(nav) − 1` daily returns with replacement from the historical return series, then compound into a NAV curve. Each simulation is independent. Fixed seed for reproducibility; `n_simulations` is user-controlled via the dashboard slider.

**p5 max drawdown:** for each simulation compute `min(curve / curve.cummax() − 1)`, then take the 5th percentile across all simulations.

---

## Streamlit Dashboard (`app.py`)

### Entry point

```bash
streamlit run app.py
```

### Structure

```python
st.set_page_config(page_title="Backtesting Framework", layout="wide")
st.title("Backtesting Framework")

tab1, tab2 = st.tabs(["Strategy Results", "Monte Carlo"])
```

Results are stored in `st.session_state` after a backtest run so the Monte Carlo tab can access the NAV series without re-running.

### Tab 1 — Strategy Results

**Controls row (single horizontal strip):**
- `st.selectbox` — Strategy: momentum / bollinger / pairs / multifactor
- `st.date_input` × 2 — Start date (default 2018-01-01), End date (default 2024-12-31)
- `st.number_input` — Initial Capital ($) (default 1,000,000)
- `st.button("▶ Run Backtest")` — triggers backtest under `st.spinner("Running…")`

**On run:** calls the same `run_strategy()` logic as the CLI (DataLoader → BacktestEngine → PerformanceAnalyzer → TearsheetGenerator). Stores `nav`, `metrics`, `spy_nav` in `st.session_state`.

**Results layout (rendered after run):**

1. **Metrics strip** — 6 `st.metric` cards in columns: Ann. Return, Sharpe, Sortino, Max Drawdown, Calmar, Hit Rate.
2. **Tearsheet** — `st.image(output_path)` displaying the PNG saved by `TearsheetGenerator.render()` to `output/<strategy>_tearsheet.png`. Note: `render()` closes the figure after saving, so `st.pyplot()` cannot be used — `st.image()` of the saved file is the correct approach.
3. **Walk-forward expander** — `st.expander("Walk-Forward Results")` collapsed by default. Contains a "▶ Run Walk-Forward Analysis" button (opening the expander alone does not trigger a Streamlit re-run). On button click, runs `WalkForwardOptimizer` under `st.spinner`, then renders `st.dataframe` of results (param columns + in-sample Sharpe + OOS Sharpe + degradation, sorted by OOS Sharpe descending). Results cached in `st.session_state` keyed by strategy name to avoid re-running.

### Tab 2 — Monte Carlo

Disabled (shows a `st.info` message) until a backtest has been run on Tab 1.

**Controls:**
- `st.slider("Simulations", 100, 5000, 1000, step=100)` — changing this value reruns the simulation.

**Results layout:**

1. **Band chart** — `st.pyplot(fig)` of a matplotlib figure: p5/p50/p95 percentile envelopes as shaded bands, actual NAV overlaid as a distinct solid line. x-axis = dates, y-axis = NAV value.
2. **Metrics strip** — 4 cards: Median terminal NAV, p5 terminal NAV, p5 max drawdown, Probability of loss.

---

## `requirements.txt` change

Add one line:
```
streamlit>=1.35
```

No Plotly. All charts use matplotlib via `st.pyplot()`.

---

## Testing

No new test files required. The three new modules are additive:
- `WalkForwardOptimizer` is exercised by running the dashboard or CLI.
- `MonteCarloSimulator` takes a `pd.Series` — trivial to unit test with a synthetic NAV if desired.
- The dashboard itself is not unit-tested (Streamlit UI testing is out of scope).

Existing 54 tests are unaffected — no strategy signatures change for callers that pass `params=None` (the default).

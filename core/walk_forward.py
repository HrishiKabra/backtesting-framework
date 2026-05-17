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

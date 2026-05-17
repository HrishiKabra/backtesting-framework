import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    def __init__(self, nav: pd.Series, risk_free_rate: float = 0.05):
        self.nav = nav
        self.rfr = risk_free_rate
        self.daily_returns = nav.pct_change().dropna()

    def compute(self) -> dict:
        r = self.daily_returns
        n_days = len(r)
        total_return = self.nav.iloc[-1] / self.nav.iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / n_days) - 1
        ann_vol = max(r.std() * np.sqrt(252), 1e-8)

        # Sortino downside deviation: RMS of min(r, 0) across all days.
        # Uses all trading days (positive return days contribute 0), matching the
        # textbook definition from Sortino & van der Meer (1991).
        downside_returns = np.minimum(r.values, 0.0)
        downside_vol = max(np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252), 1e-8)

        running_max = self.nav.cummax()
        drawdown = self.nav / running_max - 1
        max_dd = drawdown.min()

        underwater = drawdown < 0
        max_dd_duration = self._max_consecutive(underwater)

        sharpe = (ann_return - self.rfr) / ann_vol
        sortino = (ann_return - self.rfr) / downside_vol
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else (float('inf') if ann_return > 0 else 0.0)

        # Monthly returns matrix: year × month
        monthly = (
            r.resample("ME")
            .apply(lambda x: (1 + x).prod() - 1)
        )
        df = pd.DataFrame({
            "return": monthly.values,
            "year": monthly.index.year,
            "month": monthly.index.month,
        })
        matrix = df.pivot(index="year", columns="month", values="return")

        return {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": max_dd_duration,
            "cagr": ann_return,
            "calmar_ratio": calmar,
            "hit_rate": (r > 0).mean(),
            "total_return": total_return,
            "monthly_returns_matrix": matrix,
        }

    @staticmethod
    def _max_consecutive(bool_series: pd.Series) -> int:
        max_run = 0
        current_run = 0
        for val in bool_series:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

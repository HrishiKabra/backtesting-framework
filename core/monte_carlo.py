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

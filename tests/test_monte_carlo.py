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

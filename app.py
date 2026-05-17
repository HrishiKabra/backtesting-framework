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
        run_clicked = st.button("Run Backtest", use_container_width=True)

    if run_clicked:
        cfg = BacktestConfig(
            tickers=DEV_UNIVERSE,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=float(capital),
        )
        with st.spinner("Fetching data and running backtest..."):
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
            if st.button("Run Walk-Forward Analysis"):
                wfo_key = f"wfo_{sn}"
                base_params = WFO_BASE_PARAMS.get(sn, {})
                with st.spinner("Running walk-forward optimization (this takes ~2 min)..."):
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
        mc_key = f"mc_{n_sims}"
        if mc_key not in st.session_state:
            with st.spinner("Running Monte Carlo simulation..."):
                mc = MonteCarloSimulator(st.session_state["nav"], n_simulations=n_sims)
                mc.run()
                st.session_state[mc_key] = (mc.percentile_bands(), mc.summary_metrics())
        bands, mc_metrics = st.session_state[mc_key]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(
            bands.index, bands["p5"], bands["p95"],
            alpha=0.2, color="steelblue", label="p5-p95 band"
        )
        ax.fill_between(
            bands.index, bands["p5"], bands["p50"],
            alpha=0.1, color="steelblue"
        )
        ax.plot(bands.index, bands["p50"], color="steelblue", linewidth=1.0, label="p50 (median)")
        ax.plot(bands.index, bands["actual"], color="crimson", linewidth=1.5, label="Actual NAV")
        ax.set_title(f"Monte Carlo Simulation ({n_sims:,} paths) - IID Bootstrap")
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

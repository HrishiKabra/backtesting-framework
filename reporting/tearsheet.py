import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from core.performance import PerformanceAnalyzer


class TearsheetGenerator:
    def __init__(self, strategy_name: str = "Strategy"):
        self.strategy_name = strategy_name

    def render(
        self,
        metrics: dict,
        strategy_nav: pd.Series,
        spy_nav: pd.Series,
        output_path: str,
    ) -> None:
        if not strategy_nav.index.equals(spy_nav.index):
            raise ValueError(
                "strategy_nav and spy_nav must share the same index. "
                "Fetch SPY with identical date range and normalize to 1.0 at start."
            )

        daily_returns = strategy_nav.pct_change().dropna()
        drawdown = strategy_nav / strategy_nav.cummax() - 1

        # Normalize both to 1.0 at start
        norm_strat = strategy_nav / strategy_nav.iloc[0]
        norm_spy = spy_nav / spy_nav.iloc[0]

        rolling_sharpe = (
            daily_returns.rolling(252)
            .apply(lambda r: (r.mean() * 252 - 0.05) / (r.std() * np.sqrt(252) + 1e-8))
        )

        fig = plt.figure(figsize=(16, 18))
        fig.suptitle(f"{self.strategy_name} — Performance Tearsheet", fontsize=14, y=0.98)
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3)

        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(norm_strat.index, norm_strat, label=self.strategy_name, linewidth=1.5)
        ax1.plot(norm_spy.index, norm_spy, label="SPY Benchmark", linewidth=1.0, alpha=0.7)
        ax1.set_title("Cumulative Returns (normalized to 1.0)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown (underwater)
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)

        # 3. Monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, :])
        matrix = metrics.get("monthly_returns_matrix")
        if matrix is not None and not matrix.empty:
            month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            matrix.columns = [month_labels[c - 1] for c in matrix.columns if 1 <= c <= 12]
            sns.heatmap(
                matrix * 100,
                ax=ax3,
                cmap="RdYlGn",
                center=0,
                fmt=".1f",
                annot=True,
                linewidths=0.5,
                cbar_kws={"label": "Return (%)"},
            )
            ax3.set_title("Monthly Returns (%)")

        # 4. Rolling 252-day Sharpe
        ax4 = fig.add_subplot(gs[3, :])
        ax4.plot(rolling_sharpe.index, rolling_sharpe, linewidth=1.0, color="steelblue")
        ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax4.set_title("Rolling 252-Day Sharpe Ratio")
        ax4.grid(True, alpha=0.3)

        # 5. Summary stats table
        ax5 = fig.add_subplot(gs[4, :])
        ax5.axis("off")
        rows = [
            ["Annualized Return", f"{metrics['annualized_return']:.2%}"],
            ["Annualized Volatility", f"{metrics['annualized_volatility']:.2%}"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
            ["Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"],
            ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
            ["Max DD Duration", f"{metrics['max_drawdown_duration_days']} days"],
            ["CAGR", f"{metrics['cagr']:.2%}"],
            ["Calmar Ratio", f"{metrics['calmar_ratio']:.2f}"],
            ["Hit Rate", f"{metrics['hit_rate']:.2%}"],
        ]
        table = ax5.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        ax5.set_title("Performance Summary", pad=10)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Tearsheet saved to {output_path}")

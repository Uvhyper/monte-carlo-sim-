"""
Monte Carlo Graph Module
Provides an importable MonteCarloGraph class that wraps the Monte Carlo
simulation and its Tkinter/Matplotlib window from quant_analysis.py.

This module was restored from the MASTER project — it was accidentally
emptied during recent edits.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np


class MonteCarloGraph:
    """
    Thin wrapper around the Monte Carlo simulation + display logic.

    Usage (already done automatically in main.py via background thread):

        from monte_carlo_graph import MonteCarloGraph
        graph = MonteCarloGraph(df)
        window = graph.show_window()   # returns a tk.Toplevel, stays open until closed

    The simulation itself (40k paths × 252 days) is run in a background thread
    via main.py → background_compute(). This class is safe to instantiate from
    any thread; .show_window() must be called from the main thread.
    """

    def __init__(self, df, n_sims: int | None = None, horizon: int | None = None, seed: int = 42):
        """
        Parameters
        ----------
        df      : pd.DataFrame  — OHLCV data with technical indicators (from fetch_data())
        n_sims  : int | None    — override MC_SIMS (default: use quant_analysis.MC_SIMS = 40k)
        horizon : int | None    — override MC_HORIZON (default: 252 trading days)
        seed    : int           — random seed for reproducibility
        """
        from quant_analysis import MC_SIMS, MC_HORIZON
        self.df      = df
        self.n_sims  = n_sims  if n_sims  is not None else MC_SIMS
        self.horizon = horizon if horizon is not None else MC_HORIZON
        self.seed    = seed
        self.result: dict | None = None   # populated after .run()

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the Monte Carlo simulation and cache the result dict.
        Safe to call from a background thread.
        """
        from quant_analysis import monte_carlo
        self.result = monte_carlo(
            self.df,
            n_sims=self.n_sims,
            horizon=self.horizon,
            seed=self.seed,
        )
        return self.result

    # ------------------------------------------------------------------
    def show_window(self):
        """
        Open the standalone MC popup window.
        MUST be called from the main Tkinter thread (same thread as plt.show()).
        Returns the tk.Toplevel window handle.
        """
        if self.result is None:
            print("[MonteCarloGraph] No result yet — calling run() first...")
            self.run()

        from quant_analysis import show_monte_carlo_window
        return show_monte_carlo_window(self.result)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a short text summary of the simulation results."""
        if self.result is None:
            return "[MonteCarloGraph] No results — call .run() first."
        r = self.result
        return (
            f"Monte Carlo ({self.n_sims:,} sims, {self.horizon} days):\n"
            f"  Current  : ${r['last_price']:.2f}\n"
            f"  Mean     : ${r['p_mean']:.2f}\n"
            f"  Median   : ${r['p_median']:.2f}\n"
            f"  1-99 %   : ${r['perc_1']:.2f}  –  ${r['perc_99']:.2f}\n"
            f"  5-95 %   : ${r['perc_5']:.2f}  –  ${r['perc_95']:.2f}\n"
            f"  Drift    : {r['drift']:.6f}\n"
            f"  Sigma    : {r['sigma']:.6f}"
        )


# ---------------------------------------------------------------------------
# Standalone entry point — for quick testing of the MC window on its own
# ---------------------------------------------------------------------------
def run_standalone():
    """
    Run MC simulation and display window standalone (for testing).
    Example:
        python src/monte_carlo_graph.py
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from quant_analysis import fetch_data
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    print("[MonteCarloGraph] Fetching data...")
    df = fetch_data()

    mcg = MonteCarloGraph(df)
    print("[MonteCarloGraph] Running simulation...")
    mcg.run()
    print(mcg.summary())
    mcg.show_window()
    plt.show()


if __name__ == "__main__":
    run_standalone()

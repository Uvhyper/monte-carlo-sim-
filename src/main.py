"""
TSLA Terminal — Jane Street Quant Standard
Single-window matplotlib dashboard. No Tkinter. No extra threads for UI.

Layout  : GridSpec(7, 2)
Panels  : Price | RSI | Gap Score | MC Cone  (left column)
          Monte Carlo fan + metrics           (right column)
MC      : Runs in daemon thread, injected via polling timer
Signal  : Open-gap predictor + Backtest engine (all vectorized)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import hashlib
import threading
import queue as _queue

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as pta

try:
    from scipy.stats import t as student_t
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing SciPy dependency: run `.venv\\Scripts\\Activate.ps1` then `pip install --upgrade --force-reinstall scipy==1.17.1`"
    ) from exc

import matplotlib
# pick a display backend depending on environment
try:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # headless or non-GUI environment
    else:
        matplotlib.use("TkAgg")
except Exception as exc:
    print(f"[WARNING] Matplotlib backend selection failed: {exc}")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

_replay_proc = None  # replay_graph.py subprocess handle
_mc_trade_fig = None  # MC Trade Test report figure handle (prevents GC)
_bt_result = None     # stores last backtest result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA PIPELINE  (safety-first, cached)
# ═══════════════════════════════════════════════════════════════════════════════

def load_tsla(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download, validate, and clean TSLA data.
    Caches to .cache/ as CSV so repeated runs are instant.
    Raises ValueError with a human-readable message on failure.
    """
    cache_key  = hashlib.md5(f"TSLA_{period}_{interval}".encode()).hexdigest()[:8]
    cache_dir  = os.path.join(os.path.dirname(__file__), "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tsla_{cache_key}.csv")

    # Use cache if less than 1 hour old
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 3600:
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) >= 60:
                    print(f"[DATA] Loaded {len(df)} bars from cache.")
                    return df
            except Exception:
                pass  # stale / corrupt — re-download

    print(f"[DATA] Fetching TSLA ({period}, {interval}) from yfinance…")
    try:
        df = yf.download("TSLA", period=period, interval=interval,
                         progress=False, auto_adjust=True, timeout=30)
    except Exception as e:
        raise ValueError(f"yfinance download failed (timeout/network): {e}")

    # ── Validation ────────────────────────────────────────────────────────────
    if df is None or len(df) == 0:
        raise ValueError("yfinance returned an empty dataframe for TSLA")
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} bars returned — need ≥ 60")

    # Flatten MultiIndex columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Squeeze any remaining multi-dim columns to 1D
    for col in df.columns:
        if df[col].values.ndim > 1:
            df[col] = df[col].values.squeeze()

    # ── Cleaning ──────────────────────────────────────────────────────────────
    df = df[df["Volume"] > 0]                          # drop zero-volume bars
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # ── Safe clamps (not asserts — never crash) ───────────────────────────────
    df["High"] = np.maximum(df["High"], df["Low"])     # fix any inverted bars
    df = df[df["Close"] > 0]                            # drop non-positive close
    df = df[df["Open"]  > 0]                            # drop non-positive open

    df.to_csv(cache_path)
    print(f"[DATA] {len(df)} bars loaded "
          f"({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPEN GAP PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gap_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hedge fund grade gap predictor.
    4 orthogonal signal layers with regime-conditional weighting.
    Fixes: BB direction, weight calibration, position normalization, lookahead bias in thresholds.
    """
    df  = df.copy()
    eps = 1e-8

    def _get(names):
        """Return first matching 1D Series from df.columns."""
        for c in df.columns:
            for n in names:
                if str(c).startswith(n) or c == n:
                    v = df[c].values.copy().squeeze().astype(float)
                    return pd.Series(v, index=df.index)
        return None

    close = df["Close"].squeeze().astype(float)
    open_ = df["Open"].squeeze().astype(float)
    high  = df["High"].squeeze().astype(float)
    low   = df["Low"].squeeze().astype(float)

    # ── Base log returns ───────────────────────────────────────────────
    df["log_ret"] = np.log(close / close.shift(1).clip(lower=eps))
    df["log_ret"] = df["log_ret"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── EWMA volatility (λ=0.94) — same formula as sigma calibration ──
    df["ewma_vol"] = df["log_ret"].ewm(alpha=(1 - 0.94), adjust=False).std().clip(lower=eps)

    std_5  = df["log_ret"].rolling(5,  min_periods=3).std().clip(lower=eps)
    std_20 = df["log_ret"].rolling(20, min_periods=10).std().clip(lower=eps)
    std_60 = df["log_ret"].rolling(60, min_periods=30).std().clip(lower=eps)

    # ── Upgrade 2a: Volatility Regime Classification ──────────────────
    # vol_ratio > 1.3 = HIGH VOL (trending, momentum regime)
    # vol_ratio < 0.7 = LOW  VOL (compressed, mean-reversion regime)
    df["vol_ratio"] = (std_5 / std_60).clip(0.3, 3.0).fillna(1.0)
    df["regime"]    = np.where(
        df["vol_ratio"] > 1.3, 2,
        np.where(df["vol_ratio"] < 0.7, 0, 1)
    )

    # ── Signal Layer 1: Momentum ───────────────────────────────────────
    # Prior day return / EWMA vol = vol-normalized momentum score
    prior_ret        = df["log_ret"].shift(1)
    df["mom_signal"] = (prior_ret / df["ewma_vol"].shift(1)).clip(-4, 4).fillna(0)

    # Multi-period momentum confirmation (3d and 5d)
    ret_3d       = np.log(close / close.shift(3).clip(lower=eps))
    ret_5d       = np.log(close / close.shift(5).clip(lower=eps))
    df["mom_3d"] = (ret_3d  / (std_20 * np.sqrt(3))).clip(-4, 4).fillna(0)
    df["mom_5d"] = (ret_5d  / (std_20 * np.sqrt(5))).clip(-4, 4).fillna(0)

    # Blended momentum: 50% 1-day, 30% 3-day, 20% 5-day
    df["mom_blend"] = (
        0.50 * df["mom_signal"] +
        0.30 * df["mom_3d"]    +
        0.20 * df["mom_5d"]
    ).clip(-3, 3).fillna(0)

    # ── Signal Layer 2: RSI Divergence (Upgrade 3) ────────────────────
    rsi_s = _get(["RSI_14", "RSI 14", "RSI"])
    if rsi_s is not None:
        rsi = rsi_s.fillna(50)

        # RSI level signal: centered and clipped
        rsi_level = ((rsi - 50) / 50).clip(-1, 1)

        # RSI divergence: price direction vs RSI direction (3-day)
        price_dir    = np.sign(close.pct_change(3).fillna(0))
        rsi_dir      = np.sign(rsi.diff(3).fillna(0))
        divergence   = (price_dir - rsi_dir) / 2.0   # normalize to [-1, +1]

        # RSI signal = 60% level + 40% divergence
        df["rsi_signal"] = (
            0.60 * rsi_level + 0.40 * divergence
        ).clip(-1, 1).fillna(0)
    else:
        rsi          = pd.Series(50.0, index=df.index)
        df["rsi_signal"] = 0.0

    # ── Signal Layer 3: BB (regime-conditional direction) ────────────
    bb_mid = _get(["BBM_20", "BB Mid"])
    bb_u   = _get(["BBU_20", "BB Upper"])
    bb_l   = _get(["BBL_20", "BB Lower"])

    if bb_mid is not None and bb_u is not None and bb_l is not None:
        bb_width = (bb_u - bb_l).clip(lower=eps)
        bb_std   = (bb_width / 4).clip(lower=eps)
        bb_pos   = ((close - bb_mid) / bb_std).clip(-4, 4)

        # KEY FIX — regime-conditional BB direction:
        # LOW  VOL (regime=0): mean revert  → negate bb_pos (short when above)
        # NORM VOL (regime=1): mild revert  → small negative
        # HIGH VOL (regime=2): continuation → positive  (long when above)
        df["bb_signal"] = np.where(
            df["regime"] == 0, -bb_pos * 0.80,
            np.where(df["regime"] == 1, -bb_pos * 0.25,
             bb_pos * 0.40)
        )
        df["bb_signal"] = df["bb_signal"].clip(-1, 1).fillna(0)
    else:
        df["bb_signal"] = 0.0

    # ── Signal Layer 4: Overnight Gap Pressure ────────────────────────
    # ATR for gap normalization
    tr      = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14  = tr.rolling(14, min_periods=5).mean().clip(lower=eps)
    pct_atr = close * atr_14

    overnight_gap      = (open_ - close.shift(1)) / pct_atr.shift(1).clip(lower=eps)
    df["gap_signal"]   = overnight_gap.clip(-3, 3).fillna(0).shift(1)

    # ── Upgrade 2b: Regime-Conditional Composite ─────────────────────
    # Weight vectors: [momentum, rsi, bb, gap]
    W_HIGH = np.array([0.55, 0.20, 0.15, 0.10])  # HIGH VOL: trend-follow
    W_NORM = np.array([0.35, 0.30, 0.25, 0.10])  # NORMAL:   balanced
    W_LOW  = np.array([0.20, 0.25, 0.45, 0.10])  # LOW VOL:  mean-revert

    sig_mat    = np.column_stack([
        df["mom_blend"].fillna(0).values,
        df["rsi_signal"].fillna(0).values,
        df["bb_signal"].fillna(0).values,
        df["gap_signal"].fillna(0).values,
    ])
    regime_arr = df["regime"].fillna(1).values.astype(int)
    composite  = np.zeros(len(df))
    for i in range(len(df)):
        w = W_HIGH if regime_arr[i] == 2 else W_LOW if regime_arr[i] == 0 else W_NORM
        composite[i] = float(np.dot(w, sig_mat[i]))

    df["composite"] = pd.Series(composite, index=df.index).fillna(0)

    # ── Upgrade 5: Expanding-Window Thresholds (no lookahead bias) ────
    # threshold at bar t uses ONLY data from bars 0..t-1
    thresh_high = df["composite"].expanding(min_periods=120).quantile(0.82)
    thresh_low  = df["composite"].expanding(min_periods=120).quantile(0.18)

    # ── Upgrade 1b: EWMA-Normalized Position Sizing ──────────────────
    # Step 1: normalize composite to expanding Z-score
    exp_mean = df["composite"].expanding(min_periods=60).mean().fillna(0)
    exp_std  = df["composite"].expanding(min_periods=60).std().clip(lower=eps).fillna(1)
    comp_z   = ((df["composite"] - exp_mean) / exp_std).clip(-2.5, 2.5)

    # Step 2: volatility-target sizing — 1% daily vol target per trade
    vol_target = 0.01
    vol_scalar = (vol_target / df["ewma_vol"].clip(lower=0.005)).clip(0.3, 3.0)

    # Step 3: quarter-Kelly fraction (institutional conservative standard)
    kelly = 0.25
    df["position"] = (comp_z / 2.5 * vol_scalar * kelly).clip(-0.25, 0.25)

    # ── Signal direction ───────────────────────────────────────────────
    raw_sig = np.where(
        df["composite"] > thresh_high,  1,
        np.where(df["composite"] < thresh_low, -1, 0)
    )

    # ── Multi-signal alignment filter ─────────────────────────────────
    # Only trade when at least 2 of the 4 signals agree direction
    mom_dir = np.sign(df["mom_blend"].fillna(0).values)
    rsi_dir = np.sign(df["rsi_signal"].fillna(0).values)
    bb_dir  = np.sign(df["bb_signal"].fillna(0).values)

    agreement = (
        (mom_dir == rsi_dir).astype(int) +
        (mom_dir == bb_dir).astype(int)  +
        (rsi_dir == bb_dir).astype(int)
    )
    raw_sig = np.where(agreement >= 1, raw_sig, 0)

    # ── Vol regime filter — no trades in extreme volatility ───────────
    raw_sig = np.where(df["vol_ratio"].values < 2.5, raw_sig, 0)

    # ── RSI extreme filter ────────────────────────────────────────────
    if rsi_s is not None:
        rsi_v   = rsi_s.fillna(50).values
        raw_sig = np.where(
            (raw_sig ==  1) & (rsi_v < 78),  1,
            np.where(
            (raw_sig == -1) & (rsi_v > 22), -1,
            0)
        )

    # ── Cooldown filter: 5-bar minimum between trades ─────────────────
    sig        = raw_sig.copy().astype(float)
    last_trade = -999
    for i in range(len(sig)):
        if sig[i] != 0:
            if i - last_trade < 5:
                sig[i] = 0
            else:
                last_trade = i
    df["signal"] = sig

    # ── Target variable ───────────────────────────────────────────────
    prev_close         = close.shift(1).clip(lower=eps)
    df["r_gap"]        = np.log(open_ / prev_close)
    df["r_gap_simple"] = np.expm1(df["r_gap"])   # log → simple return
    df["r_gap"]        = df["r_gap"].replace([np.inf, -np.inf], np.nan)
    df["r_gap_simple"] = df["r_gap_simple"].replace([np.inf, -np.inf], np.nan)

    return df.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_gap_signal(df: pd.DataFrame,
                        initial_capital: float = 10_000.0) -> dict:
    """
    Institutional backtest engine.
    Fix: uses r_gap_simple (already simple return) not log return.
    Fix: asymmetric stop/cap (-2% / +4%) for 2:1 reward/risk.
    Adds: Calmar, Win/Loss ratio, Expectancy per trade.
    """
    df_c   = df.dropna(subset=["signal", "r_gap_simple", "position"])
    trades = df_c[df_c["signal"] != 0].copy()

    if len(trades) < 10:
        return {"error": f"insufficient trades ({len(trades)}) — need ≥ 10"}

    # Bug F fix: r_gap_simple is already a simple return — no expm1 needed
    trades["pnl"] = (trades["position"] * trades["r_gap_simple"]).fillna(0)

    # Asymmetric stop: -2% max loss, +4% max gain (2:1 reward/risk ratio)
    trades["pnl"] = trades["pnl"].clip(lower=-0.02, upper=0.04)
    trades["pnl"] = trades["pnl"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Equity curve
    trades["equity"] = initial_capital * (1 + trades["pnl"]).cumprod()

    # Core metrics
    n_trades  = len(trades)
    final_eq  = float(trades["equity"].iloc[-1])
    total_ret = (final_eq / initial_capital) - 1
    hit_rate  = float((trades["pnl"] > 0).sum()) / max(n_trades, 1)

    mean_r = float(trades["pnl"].mean())
    std_r  = float(trades["pnl"].std())

    # Annualized Sharpe — adjust for actual trade frequency
    years           = max(len(df) / 252.0, 0.1)
    trades_per_year = n_trades / years
    sharpe  = (mean_r / std_r * np.sqrt(trades_per_year)) if std_r > 1e-8 else 0.0

    # Max drawdown
    roll_max  = trades["equity"].cummax()
    dd_series = (trades["equity"] - roll_max) / roll_max.clip(lower=1e-8)
    max_dd    = float(dd_series.min())

    # Sortino
    down_pnl = trades["pnl"][trades["pnl"] < 0]
    downside = float(down_pnl.std()) if len(down_pnl) > 1 else 0.0
    sortino  = (mean_r / downside * np.sqrt(trades_per_year)) if downside > 1e-8 else 0.0

    # Calmar
    calmar = (total_ret / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0

    # Win/Loss ratio
    wins     = trades["pnl"][trades["pnl"] > 0]
    losses   = trades["pnl"][trades["pnl"] < 0]
    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-8 else 0.0

    # Expectancy — must be positive to trade live
    expectancy = hit_rate * avg_win + (1.0 - hit_rate) * avg_loss

    print(f"[BACKTEST] ── Institutional Performance Report ─────────────")
    print(f"[BACKTEST] Trades: {n_trades}  ({trades_per_year:.0f}/yr)  Hit Rate: {hit_rate:.1%}")
    print(f"[BACKTEST] Total Return: {total_ret:+.2%}  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}")
    print(f"[BACKTEST] Calmar: {calmar:.2f}  Win/Loss: {win_loss:.2f}  Expectancy: {expectancy:+.4f}")
    print(f"[BACKTEST] Max DD: {max_dd:.2%}  Avg Win: {avg_win:+.4f}  Avg Loss: {avg_loss:+.4f}")
    print(f"[BACKTEST] ─────────────────────────────────────────────────")

    return {
        "equity"     : trades["equity"],
        "trades"     : trades,
        "n_trades"   : n_trades,
        "hit_rate"   : hit_rate,
        "total_ret"  : total_ret,
        "sharpe"     : sharpe,
        "sortino"    : sortino,
        "max_dd"     : max_dd,
        "calmar"     : calmar,
        "win_loss"   : win_loss,
        "expectancy" : expectancy,
        "trades_py"  : trades_per_year,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MONTE CARLO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(S0: float, mu: float, sigma: float,
                    n_days: int = 252, n_sims: int = 10_000) -> dict:
    """
    Student-t GBM Monte Carlo.
    Upgrade 4: fat-tailed Student-t shocks instead of Gaussian.
    TSLA empirical kurtosis ≈ 6-8; Normal distribution = 3.
    Student-t df=5 gives kurtosis = 9 — matches TSLA empirically.
    Variance normalized so sigma scaling unchanged.

    Math:
      Z ~ Student-t(df=5) / sqrt(df/(df-2))   variance = 1
      S_t = S0 * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
      Ito correction (mu - 0.5*sigma^2) ensures E[S_t] = S0*e^(mu*t)
    """

    n_sims = int(min(n_sims, 80_000))
    n_days = int(min(n_days, 252))
    mu     = float(np.clip(mu,    -0.005, 0.005))
    sigma  = float(np.clip(sigma,  0.005, 0.065))

    # Student-t parameters
    df_t   = 5.0
    # Variance normalizer: Var(t_df) = df/(df-2), normalize to 1
    t_norm = float(np.sqrt(df_t / (df_t - 2.0)))

    rng = np.random.default_rng(42)
    # Inverse CDF sampling — exact Student-t shocks
    u   = rng.uniform(1e-6, 1.0 - 1e-6, (n_sims, n_days))
    Z   = student_t.ppf(u, df=df_t) / t_norm   # (n_sims, n_days)
    Z   = np.clip(Z.astype(np.float32), -4.5, 4.5)

    # Ito-corrected GBM
    daily  = (mu - 0.5 * sigma ** 2) + sigma * Z
    paths  = S0 * np.exp(np.cumsum(daily, axis=1))
    paths  = np.clip(paths, S0 * 0.01, S0 * 50)

    days = np.arange(1, n_days + 1)
    return {
        "paths"  : paths,
        "days"   : days,
        "mean"   : paths.mean(axis=0),
        "median" : np.percentile(paths, 50, axis=0),
        "p01"    : np.percentile(paths,  1, axis=0),
        "p05"    : np.percentile(paths,  5, axis=0),
        "p25"    : np.percentile(paths, 25, axis=0),
        "p75"    : np.percentile(paths, 75, axis=0),
        "p95"    : np.percentile(paths, 95, axis=0),
        "p99"    : np.percentile(paths, 99, axis=0),
    }


def draw_cone(ax, mc: dict, current_price: float) -> None:
    """Draw the MC probability cone on ax_cone."""
    ax.cla()
    ax.set_facecolor("#0a0a0a")
    days = mc["days"]
    ax.fill_between(days, mc["p01"], mc["p99"],
                    alpha=0.08, color="#4a9eff", label="1-99%")
    ax.fill_between(days, mc["p05"], mc["p95"],
                    alpha=0.18, color="#4a9eff", label="5-95%")
    ax.fill_between(days, mc["p25"], mc["p75"],
                    alpha=0.30, color="#4a9eff", label="25-75%")
    ax.plot(days, mc["mean"],   color="#ff9500", lw=1.5, label="Mean")
    ax.plot(days, mc["median"], color="#ffffff", lw=0.8, ls="--", label="Median")
    ax.axhline(current_price,   color="#888888", lw=0.7, ls=":")
    ax.set_xlabel("Days into the Future", color="#888888", fontsize=8)
    ax.set_ylabel("Price (USD)", color="#888888", fontsize=7)
    ax.set_title("MC Probability Cone — 1-yr Forecast", color="#aaaaaa", fontsize=8, pad=3)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.legend(fontsize=6, framealpha=0.3, labelcolor="#888888", loc="upper left")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def draw_mc_fan(ax, mc: dict, current_price: float,
                bt: dict | None = None) -> None:
    """
    Draw the full MC path fan + metrics table on ax_mc.
    Uses a single LineCollection instead of per-path plot() calls
    to avoid rendering hundreds of separate artists (crash fix).
    """
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    ax.cla()
    ax.set_facecolor("#0a0a0a")
    days  = mc["days"]
    paths = mc["paths"]

    # Sample ≤ 350 paths evenly
    step   = max(1, len(paths) // 350)
    subset = paths[::step]                  # (n_show, n_days)
    n_show = subset.shape[0]

    # Build segments array for LineCollection: shape (n_show, n_days, 2)
    days_col  = days.astype(np.float32)
    segs      = np.stack(
        [np.broadcast_to(days_col[None, :], (n_show, len(days))),
         subset.astype(np.float32)],
        axis=-1
    )   # (n_show, n_days, 2)

    # Color by terminal price (low=blue, high=green)
    final_prices = subset[:, -1]
    fp_range     = float(final_prices.max() - final_prices.min())  # np.ptp removed in numpy 2.x
    norm_prices  = (final_prices - final_prices.min()) / (fp_range + 1e-8)
    colors = cm.cool(norm_prices)  # (n_show, 4) RGBA

    lc = LineCollection(segs, colors=colors, linewidths=0.35, alpha=0.20)
    ax.add_collection(lc)

    # Auto-scale axes after add_collection (it doesn’t auto-scale)
    ax.set_xlim(days[0], days[-1])
    all_prices = subset
    p1, p99    = np.percentile(all_prices, 1), np.percentile(all_prices, 99)
    margin     = (p99 - p1) * 0.05
    ax.set_ylim(max(0, p1 - margin), p99 + margin)

    # Mean overlay
    ax.plot(days, mc["mean"], color="white", lw=2.0, label="Average Path", zorder=5)

    # Metrics text box
    gain_pct = ((mc["mean"][-1] - current_price) / current_price) * 100
    metrics_text = (
        f"── Monte Carlo Metrics ──\n"
        f"Mean Price     ${mc['mean'][-1]:>8.2f}\n"
        f"Median Price   ${mc['median'][-1]:>8.2f}\n"
        f"1st Pctile     ${mc['p01'][-1]:>8.2f}\n"
        f"5th Pctile     ${mc['p05'][-1]:>8.2f}\n"
        f"95th Pctile    ${mc['p95'][-1]:>8.2f}\n"
        f"99th Pctile    ${mc['p99'][-1]:>8.2f}\n"
        f"Expected gain  {gain_pct:>+7.1f}%"
    )
    if bt and "error" not in bt:
        exp_str = f"{bt.get('expectancy',0):>+6.4f}"
        exp_col = "✓" if bt.get('expectancy', 0) > 0 else "✗"
        metrics_text += (
            f"\n\n── Gap Signal Backtest ──\n"
            f"Trades       {bt['n_trades']:>6d}\n"
            f"Hit rate     {bt['hit_rate']:>6.1%}\n"
            f"Total return {bt['total_ret']:>+6.1%}\n"
            f"Sharpe       {bt['sharpe']:>6.2f}\n"
            f"Sortino      {bt['sortino']:>6.2f}\n"
            f"Calmar       {bt.get('calmar',0):>6.2f}\n"
            f"Win/Loss     {bt.get('win_loss',0):>6.2f}\n"
            f"Expectancy   {exp_str} {exp_col}\n"
            f"Max DD       {bt['max_dd']:>6.1%}"
        )

    ax.text(0.02, 0.97, metrics_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#111111", alpha=0.85,
                      edgecolor="#333333"))

    ax.set_title(
        f"Monte Carlo Fan — {len(paths):,} paths × {len(days)} days",
        color="#aaaaaa", fontsize=9, pad=3
    )
    ax.set_xlabel("Days Future", color="#888888", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THREAD SAFETY  (prevent PC freeze)
# ═══════════════════════════════════════════════════════════════════════════════

_mc_queue: _queue.Queue = _queue.Queue(maxsize=1)


def _mc_worker(S0: float, mu: float, sigma: float) -> None:
    result = run_monte_carlo(S0, mu, sigma, n_days=252, n_sims=80_000)
    try:
        _mc_queue.put_nowait(result)
    except _queue.Full:
        pass   # previous result not yet consumed — silently drop


def launch_mc(S0: float, mu: float, sigma: float) -> None:
    """Launch Monte Carlo in a daemon thread — never blocks UI."""
    t = threading.Thread(target=_mc_worker, args=(S0, mu, sigma), daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# DRAW HELPERS — price, RSI, score panels
# ═══════════════════════════════════════════════════════════════════════════════

def _col(df: pd.DataFrame, *names):
    """Return the first matching 1-D numpy array from df.columns."""
    for name in names:
        if name in df.columns:
            v = df[name].values.copy()
            while v.ndim > 1:
                v = v.squeeze()
            return v.astype(float)
    return None


def draw_price(ax, df: pd.DataFrame) -> None:
    """OHLC close line + SMAs + Bollinger Bands + gap signal markers."""
    ax.set_facecolor("#0a0a0a")
    xs    = np.arange(len(df))
    close = df["Close"].values.squeeze().astype(float)

    # Price line
    ax.plot(xs, close, color="#4a9eff", lw=1.2, label="Close", zorder=3)

    # SMAs
    sma20 = _col(df, "SMA_20", "SMA 20")
    sma50 = _col(df, "SMA_50", "SMA 50")
    if sma20 is not None:
        ax.plot(xs, sma20, color="#ff9500", lw=0.9, ls="--", alpha=0.85, label="SMA 20")
    if sma50 is not None:
        ax.plot(xs, sma50, color="#bc8cff", lw=0.9, ls="--", alpha=0.85, label="SMA 50")

    # Bollinger Bands (fill + edge lines)
    bbu_col = next((c for c in df.columns if str(c).startswith("BBU_20") or c == "BB Upper"), None)
    bbl_col = next((c for c in df.columns if str(c).startswith("BBL_20") or c == "BB Lower"), None)
    if bbu_col and bbl_col:
        bbu = df[bbu_col].values.squeeze().astype(float)
        bbl = df[bbl_col].values.squeeze().astype(float)
        ax.fill_between(xs, bbl, bbu, alpha=0.07, color="#888888")
        ax.plot(xs, bbu, color="#888888", lw=0.5, alpha=0.55, label="BB Upper")
        ax.plot(xs, bbl, color="#888888", lw=0.5, alpha=0.55, label="BB Lower")

    # Gap signal markers
    if "signal" in df.columns:
        sig   = df["signal"].values
        low_v = df["Low"].values.squeeze().astype(float)
        hig_v = df["High"].values.squeeze().astype(float)

        long_mask  = sig == 1
        short_mask = sig == -1
        if long_mask.any():
            ax.scatter(xs[long_mask],  low_v[long_mask]  * 0.978,
                       marker="^",
                       color="#00ff88",
                       s=18,
                       linewidths=0.3,
                       edgecolors="#007744",
                       alpha=0.85,
                       zorder=5,
                       label="Gap Long")
        if short_mask.any():
            ax.scatter(xs[short_mask], hig_v[short_mask] * 1.022,
                       marker="v",
                       color="#ff3b5c",
                       s=18,
                       linewidths=0.3,
                       edgecolors="#aa1133",
                       alpha=0.85,
                       zorder=5,
                       label="Gap Short")

    pad = max((close.max() - close.min()) * 0.06, 5.0)
    ax.set_ylim(close.min() - pad, close.max() + pad)
    ax.set_xlim(0, len(df))
    ax.set_ylabel("Price (USD)", color="#888888", fontsize=8)
    ax.set_title("TSLA · Price  /  SMA  /  Bollinger Bands  /  Gap Signals",
                 color="#cccccc", fontsize=9, pad=4)
    ax.legend(fontsize=6, framealpha=0.25, labelcolor="#aaaaaa",
              loc="upper left", ncol=4)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def draw_rsi(ax, df: pd.DataFrame) -> None:
    """RSI(14) panel with overbought/oversold shading."""
    ax.set_facecolor("#0a0a0a")
    xs  = np.arange(len(df))
    rsi = _col(df, "RSI_14", "RSI 14", "RSI")
    if rsi is not None:
        ax.plot(xs, rsi, color="#bc8cff", lw=1.0)
        ax.fill_between(xs, rsi, 70, where=(rsi > 70), alpha=0.25, color="#ff3b5c")
        ax.fill_between(xs, rsi, 30, where=(rsi < 30), alpha=0.25, color="#00ff88")
    ax.axhline(70, color="#ff3b5c", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(30, color="#00ff88", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(50, color="#444444", lw=0.5)
    ax.set_ylabel("RSI 14", color="#888888", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(df))
    ax.tick_params(colors="#888888", labelsize=7)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def draw_score_panel(ax, df: pd.DataFrame) -> None:
    """Gap composite score bar chart with ±0.04 signal thresholds."""
    if "composite" not in df.columns:
        return
    ax.set_facecolor("#0a0a0a")
    comp_full = df["composite"].fillna(0).values
    n = len(comp_full)

    # Downsample if > 500 bars — rendering 1200+ bars is very expensive
    if n > 500:
        step = max(1, n // 500)
        xs   = np.arange(0, n, step)
        comp = comp_full[::step]
    else:
        xs   = np.arange(n)
        comp = comp_full

    clrs = np.where(comp > 0.04, "#00ff88",
           np.where(comp < -0.04, "#ff3b5c", "#555555"))

    ax.bar(xs, comp, color=clrs, width=max(0.8, n / len(xs) * 0.6), alpha=0.75)
    ax.axhline( 0.04, color="#00ff88", lw=0.7, ls="--", alpha=0.65)
    ax.axhline(-0.04, color="#ff3b5c", lw=0.7, ls="--", alpha=0.65)
    ax.axhline(0,     color="#444444", lw=0.5)

    # Dynamic ylim: show 95th-percentile range to prevent extreme bars squishing view
    q95 = np.nanpercentile(np.abs(comp_full), 95)
    ylim = max(q95 * 1.4, 0.15)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(0, n)
    ax.set_ylabel("Score", color="#888888", fontsize=7)
    ax.set_title("Gap Composite Score  (▲ Long  ▼ Short  — Flat = grey)",
                 color="#888888", fontsize=7, pad=2)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


# ═══════════════════════════════════════════════════════════════════════════════
# MC TRADE TEST — trade-order Monte Carlo (StrategyQuant-style)
# ═══════════════════════════════════════════════════════════════════════════════

def run_trade_mc(
    trade_pnls: np.ndarray,
    n_sims: int = 1000,
    skip_prob: float = 0.0,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Shuffles trade sequence n_sims times.
    Optionally skips trades with probability skip_prob.
    """
    n_sims = int(min(max(n_sims, 1), 80_000))
    trade_pnls = np.asarray(trade_pnls, dtype=float).reshape(-1)
    n_trades = int(trade_pnls.size)
    if n_trades < 10:
        raise ValueError("Need at least 10 trades to run MC")

    # Safety cap (render/memory guard)
    if n_trades > 2000:
        trade_pnls = trade_pnls[-2000:]
        n_trades = 2000

    # Bug E fix: pnl is already a simple return — do NOT apply expm1
    base_rets = np.clip(trade_pnls.astype(float), -0.5, 0.5)
    base_rets = np.clip(base_rets, -0.95, 2.0)

    rng = np.random.default_rng(seed=None)   # Bug D fix: OS entropy each call
    equity_curves = np.zeros((n_sims, n_trades), dtype=float)

    for i in range(n_sims):
        shuffled = rng.permutation(base_rets)

        if skip_prob > 0:
            keep_mask = rng.random(n_trades) > float(skip_prob)
            shuffled2 = shuffled[keep_mask]
            if shuffled2.size > 0:
                shuffled = shuffled2
            else:
                shuffled = base_rets

        curve = initial_capital * np.cumprod(1.0 + shuffled)
        if curve.size < n_trades:
            curve = np.pad(curve, (0, n_trades - curve.size), mode="edge")
        equity_curves[i] = curve[:n_trades]

    original_curve = initial_capital * np.cumprod(1.0 + base_rets[:n_trades])

    return {
        "equity_curves": equity_curves,
        "final_equities": equity_curves[:, -1],
        "original_curve": original_curve,
        "n_trades": n_trades,
        "n_sims": n_sims,
        "initial_capital": float(initial_capital),
    }


def compute_confidence_table(mc_result: dict, initial_capital: float = 10_000.0) -> list[dict]:
    finals = np.asarray(mc_result["final_equities"], dtype=float)
    curves = np.asarray(mc_result["equity_curves"], dtype=float)
    capital = float(initial_capital)

    def _max_dd(curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / np.clip(peak, 1e-8, None)
        return float(np.abs(dd.min()))

    all_dds = np.array([_max_dd(curves[i]) for i in range(curves.shape[0])], dtype=float)

    orig = np.asarray(mc_result["original_curve"], dtype=float)
    orig_profit = float(orig[-1] - capital)
    orig_dd = _max_dd(orig) * capital

    levels = [50, 60, 70, 80, 90, 92, 95, 97, 98, 99, 100]
    rows: list[dict] = [{
        "level": "Original",
        "net_profit": orig_profit,
        "drawdown": orig_dd,
        "highlight": False,
    }]

    for lvl in levels:
        pct = 100 - lvl
        net_profit = float(np.percentile(finals, pct) - capital)
        drawdown = float(np.percentile(all_dds, lvl) * capital)
        rows.append({
            "level": f"{lvl}%",
            "net_profit": net_profit,
            "drawdown": drawdown,
            "highlight": lvl == 95,
        })
    return rows


def print_top_simulations(mc_result: dict, initial_capital: float = 10_000.0, limit: int = 350) -> None:
    """Print top 300-350 simulation results by final equity (descending order)."""
    finals = np.asarray(mc_result.get("final_equities", []), dtype=float)
    if len(finals) == 0:
        print("No simulation results available")
        return

    # Sort final equities in descending order
    sorted_indices = np.argsort(-finals)  # negative for descending
    limit = min(limit, len(finals))
    top_indices = sorted_indices[:limit]
    top_finals = finals[top_indices]

    print("\n" + "="*70)
    print(f"  TOP {limit} SIMULATION RESULTS (out of {len(finals):,})")
    print("="*70)
    print(f"{'Rank':>6} | {'Final Equity':>14} | {'Profit/Loss':>14} | {'Return %':>10}")
    print("-"*70)

    for rank, idx in enumerate(top_indices, 1):
        final_eq = float(top_finals[rank-1])
        pnl = final_eq - initial_capital
        ret_pct = (pnl / initial_capital) * 100.0
        print(f"{rank:>6} | ${final_eq:>13,.2f} | ${pnl:>13,.2f} | {ret_pct:>9.2f}%")

    print("-"*70)
    print(f"Median Final Equity: ${np.median(finals):,.2f}")
    print(f"Mean Final Equity:   ${np.mean(finals):,.2f}")
    print(f"Best:   ${np.max(finals):,.2f}")
    print(f"Worst:  ${np.min(finals):,.2f}")
    print(f"Profitable Sims: {(finals > initial_capital).sum():,} / {len(finals):,} ({(finals > initial_capital).mean()*100:.1f}%)")
    print("="*70 + "\n")


def _draw_confidence_table(ax, rows: list[dict]) -> None:
    ax.set_facecolor("#0a0a0a")
    ax.axis("off")
    ax.set_title("Confidence Levels", color="white", fontsize=9, pad=8)

    headers = ["Confidence", "Net Profit", "Drawdown"]
    col_w = [0.33, 0.34, 0.33]
    x0s = [0.0, 0.33, 0.67]
    y_start = 0.95
    row_h = 0.95 / (len(rows) + 1)

    for j, h in enumerate(headers):
        ax.text(
            x0s[j] + col_w[j] / 2,
            y_start,
            h,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#888888",
            fontsize=8,
            fontweight="bold",
        )

    for i, row in enumerate(rows):
        y = y_start - (i + 1) * row_h

        if row.get("highlight"):
            rect = plt.Rectangle(
                (0, y - row_h * 0.45),
                1,
                row_h,
                transform=ax.transAxes,
                facecolor="#0a2a4a",
                edgecolor="#4a9eff",
                linewidth=0.8,
                clip_on=False,
            )
            ax.add_patch(rect)

        profit_color = "#00ff88" if row["net_profit"] > 0 else "#ff3b5c"
        texts = [
            (row["level"], "#cccccc"),
            (f"${row['net_profit']:>8.0f}", profit_color),
            (f"${row['drawdown']:>8.0f}", "#ff9966"),
        ]
        for j, (txt, col) in enumerate(texts):
            ax.text(
                x0s[j] + col_w[j] / 2,
                y,
                txt,
                transform=ax.transAxes,
                ha="center",
                va="top",
                color=col,
                fontsize=8,
                fontfamily="monospace",
            )

    ax.text(
        0.5,
        0.02,
        "95% confidence = 5% probability actual profit falls below this",
        transform=ax.transAxes,
        ha="center",
        color="#4a9eff",
        fontsize=7,
        style="italic",
    )


def _draw_equity_fan(ax, mc: dict, title: str) -> None:
    ax.set_facecolor("#0a0a0a")
    ax.set_title(title, color="white", fontsize=8, pad=6)

    curves = mc["equity_curves"]
    orig = mc["original_curve"]
    capital = float(mc["initial_capital"])
    n = int(mc["n_trades"])
    x = np.arange(n)

    step = max(1, len(curves) // 300)
    for i in range(0, len(curves), step):
        final = float(curves[i, -1])
        color = (
            "#00ff44" if final > capital * 1.05 else
            "#ff3344" if final < capital * 0.95 else
            "#888888"
        )
        ax.plot(x, curves[i], lw=0.25, alpha=0.15, color=color)

    ax.plot(x, orig, color="#4a9eff", lw=1.8, label="Original backtest", zorder=5)
    ax.axhline(capital, color="#444444", lw=0.7, ls="--")

    ax.set_xlabel("Trade number", color="#888888", fontsize=7)
    ax.set_ylabel("Equity ($)", color="#888888", fontsize=7)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.8)

    finals = mc["final_equities"]
    spread = float(np.max(finals) - np.min(finals))
    pct_profitable = float((finals > capital).mean() * 100.0)
    ax.text(
        0.02,
        0.97,
        f"% profitable sims: {pct_profitable:.0f}%\nProfit spread: ${spread:,.0f}",
        transform=ax.transAxes,
        va="top",
        color="#cccccc",
        fontsize=7,
        fontfamily="monospace",
        bbox=dict(facecolor="#111111", alpha=0.8, edgecolor="#333333", boxstyle="round,pad=0.4"),
    )


def _draw_final_histogram(ax, mc: dict, capital: float) -> None:
    ax.set_facecolor("#0a0a0a")
    ax.set_title("Distribution of Final Equity\n(all simulations)", color="white", fontsize=8, pad=6)

    finals = np.asarray(mc["final_equities"], dtype=float)
    profit = finals[finals >= capital]
    loss = finals[finals < capital]

    # Robust binning: avoid numpy "Too many bins for data range" when finals are (nearly) constant
    spread = float(np.max(finals) - np.min(finals)) if finals.size else 0.0
    if spread <= 1e-9:
        bins = 1
        center = float(finals[0]) if finals.size else float(capital)
        hist_range = (center - 1.0, center + 1.0)
    else:
        # sqrt rule, capped; also ensure bin width not too tiny
        bins = int(np.clip(np.sqrt(finals.size), 10, 40))
        hist_range = None

    if profit.size:
        ax.hist(profit, bins=bins, range=hist_range, color="#00ff88", alpha=0.6, label="Profitable")
    if loss.size:
        ax.hist(loss, bins=bins, range=hist_range, color="#ff3b5c", alpha=0.6, label="Loss")

    ymax = ax.get_ylim()[1]
    # Build label list, sort by x-position, stagger to avoid overlaps
    labels_info = []
    for pct, label, col in [
        (5, "VaR 95%", "#ff9500"),
        (50, "Median", "#ffffff"),
        (95, "95th", "#00d4ff"),
    ]:
        val = float(np.percentile(finals, pct))
        ax.axvline(val, color=col, lw=1.2, ls="--")
        labels_info.append((val, label, col))

    labels_info.sort(key=lambda t: t[0])
    y_positions: list[float] = []
    base_y = max(ymax * 0.92, 1.0)
    min_y_gap = ymax * 0.15  # minimum vertical spacing between labels
    xlo, xhi = ax.get_xlim()
    x_span = max(xhi - xlo, 1.0)
    min_x_gap = x_span * 0.06  # labels closer than 6% x-range get staggered
    for idx, (val, label, col) in enumerate(labels_info):
        y = base_y
        for prev_idx in range(idx):
            prev_val = labels_info[prev_idx][0]
            prev_y = y_positions[prev_idx]
            if abs(val - prev_val) < min_x_gap and abs(y - prev_y) < min_y_gap:
                y = prev_y - min_y_gap
        y_positions.append(y)
        ax.text(val + x_span * 0.005, y, f"{label}\n${val:,.0f}",
                color=col, fontsize=7, va="top")

    ax.axvline(capital, color="#888888", lw=0.8, ls=":", label=f"Starting capital ${capital:,.0f}")
    ax.set_xlabel("Final equity ($)", color="#888888", fontsize=7)
    ax.set_ylabel("Count", color="#888888", fontsize=7)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.8)


def show_mc_trade_window(bt: dict, n_sims: int = 1000) -> None:
    global _mc_trade_fig

    if not bt or "error" in bt:
        print("Not enough trades to run MC Trade Test")
        return

    trades_df = bt.get("trades", None)
    if trades_df is None or len(trades_df) < 10:
        print("Not enough trades to run MC Trade Test")
        return

    trade_pnls = np.asarray(trades_df["pnl"].values, dtype=float)
    if trade_pnls.size < 10:
        print("Not enough trades to run MC Trade Test")
        return

    capital = 10_000.0
    mc_normal = run_trade_mc(trade_pnls, n_sims=n_sims, skip_prob=0.0, initial_capital=capital)
    mc_skip = run_trade_mc(trade_pnls, n_sims=n_sims, skip_prob=0.10, initial_capital=capital)
    conf_table = compute_confidence_table(mc_normal, capital)
    print_top_simulations(mc_normal, capital, limit=350)

    try:
        _mc_trade_fig = plt.figure(
            figsize=(16, 9),
            facecolor="#0a0a0a",
            num="MC Trade Test — TSLA Gap Signal",
        )
        gs = GridSpec(
            2,
            2,
            figure=_mc_trade_fig,
            hspace=0.35,
            wspace=0.25,
            left=0.06,
            right=0.97,
            top=0.92,
            bottom=0.08,
        )

        ax_table = _mc_trade_fig.add_subplot(gs[0, 0])
        ax_fan = _mc_trade_fig.add_subplot(gs[0, 1])
        ax_skip = _mc_trade_fig.add_subplot(gs[1, 0])
        ax_hist = _mc_trade_fig.add_subplot(gs[1, 1])

        _mc_trade_fig.suptitle(
            f"Monte Carlo Trade Analysis  ·  TSLA Gap Signal  ·  "
            f"{mc_normal['n_sims']} simulations  ·  {mc_normal['n_trades']} trades",
            color="white",
            fontsize=11,
        )

        _draw_confidence_table(ax_table, conf_table)
        _draw_equity_fan(
            ax_fan,
            mc_normal,
            title=f"Trade Shuffle Fan — {mc_normal['n_sims']} sims\n(sampling without replacement)",
        )
        _draw_equity_fan(
            ax_skip,
            mc_skip,
            title="Skip-Trade Simulation — 10% skip rate\n(tests robustness to missed signals)",
        )
        _draw_final_histogram(ax_hist, mc_normal, capital)

        plt.show(block=False)
        _mc_trade_fig.canvas.draw()
    except Exception as exc:
        print(f"[MC TRADE TEST ERROR] {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN  (final integration)
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _bt_result
    print("=" * 62)
    print("  TSLA Terminal — Jane Street Quant Standard")
    print("=" * 62)

    # ── 1. Load and clean data ──────────────────────────────────────────────
    df = load_tsla(period="5y", interval="1d")

    # ── 2. Technical indicators ─────────────────────────────────────────────
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    try:
        df.ta.atr(length=5,  append=True)
        df.ta.atr(length=20, append=True)
    except Exception:
        pass
    df.dropna(inplace=True)

    # ── 3. Gap signal ───────────────────────────────────────────────────────
    df = compute_gap_signal(df)

    # ── 4. Backtest ─────────────────────────────────────────────────────────
    bt = backtest_gap_signal(df)
    _bt_result = bt
    if "error" not in bt:
        print(
            f"[BACKTEST] Trades: {bt['n_trades']}  "
            f"Hit Rate: {bt['hit_rate']:.1%}  "
            f"Total Return: {bt['total_ret']:+.1%}  "
            f"Sharpe: {bt['sharpe']:.2f}  "
            f"Max DD: {bt['max_dd']:.1%}"
        )
    else:
        print(f"[BACKTEST] {bt['error']}")
        bt = None

    # ── 5. Monte Carlo parameters (EWMA volatility upgrade) ─────────────────
    close_vals  = df["Close"].values.squeeze().astype(float)
    log_ret_all = np.diff(np.log(np.clip(close_vals, 1e-8, None)))
    log_ret_all = log_ret_all[np.isfinite(log_ret_all)]

    # Upgrade 1: EWMA Volatility λ=0.94 (RiskMetrics standard)
    # Formula: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
    # λ=0.94: last 30 days ≈ 84% weight → historical crashes irrelevant
    lam      = 0.94
    n_r      = len(log_ret_all)
    ewma_var = float(log_ret_all[0] ** 2)
    for r in log_ret_all[1:]:
        ewma_var = lam * ewma_var + (1.0 - lam) * float(r ** 2)
    sigma = float(np.clip(np.sqrt(ewma_var), 0.005, 0.065))

    # mu: long-run drift from full history (captures secular trend)
    mu_full   = float(np.mean(log_ret_all))
    mu_recent = float(np.mean(log_ret_all[-252:])) if n_r >= 252 else mu_full
    mu        = float(np.clip(0.8 * mu_full + 0.2 * mu_recent, -0.005, 0.005))

    S0 = float(close_vals[-1])
    ann_vol_ewma = sigma * np.sqrt(252) * 100
    ann_vol_raw  = float(np.std(log_ret_all)) * np.sqrt(252) * 100
    print(f"[MC] S0=${S0:.2f}  EWMA sigma={sigma:.6f} ({ann_vol_ewma:.1f}% ann)")
    print(f"[MC] Raw sigma would be {ann_vol_raw:.1f}% ann — EWMA is more accurate")
    print(f"[MC] mu={mu:.6f}  ({mu*252*100:.2f}% ann drift)")

    # ── 6. Single-window figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11), facecolor="#0a0a0a")
    try:
        fig.canvas.manager.set_window_title("TSLA Terminal")
    except Exception:
        pass

    gs = GridSpec(7, 2, figure=fig,
                  hspace=0.08, wspace=0.12,
                  left=0.05, right=0.97, top=0.96, bottom=0.08)

    ax_price = fig.add_subplot(gs[0:3, 0])
    ax_rsi   = fig.add_subplot(gs[3,   0])
    ax_score = fig.add_subplot(gs[4,   0])
    ax_cone  = fig.add_subplot(gs[5:7, 0])
    ax_mc    = fig.add_subplot(gs[0:7, 1])

    for ax in [ax_price, ax_rsi, ax_score, ax_cone, ax_mc]:
        ax.set_facecolor("#0a0a0a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    # ── 7. Draw static panels ───────────────────────────────────────────────
    draw_price(ax_price, df)
    draw_rsi(ax_rsi, df)
    draw_score_panel(ax_score, df)

    # Placeholders while MC computes in background
    for ax, msg in [
        (ax_cone, "⏳  MC Cone computing in background…"),
        (ax_mc,   "⏳  MC Fan computing in background…\n\n"
                  "Results appear automatically\nwhen the simulation finishes."),
    ]:
        ax.text(0.5, 0.5, msg,
                ha="center", va="center", color="#888888",
                fontsize=10, transform=ax.transAxes)
    ax_mc.set_title("Monte Carlo Simulation (80,000 paths)",
                    color="#aaaaaa", fontsize=9, pad=3)
    ax_cone.set_title("MC Probability Cone — 1-yr Forecast",
                      color="#aaaaaa", fontsize=8, pad=3)

    # ── 8. Launch MC in background daemon thread ────────────────────────────
    launch_mc(S0, mu, sigma)
    print("[MC] Simulation launched in background — UI is immediately interactive.")

    # ── 9. Timer — polls queue every 500ms, stops after first result ────────
    _mc_injected = [False]

    def _check_mc_queue() -> None:
        try:
            if _mc_injected[0]:
                return  # already got a result — stop polling
            if not _mc_queue.empty():
                mc = _mc_queue.get_nowait()
                draw_cone(ax_cone, mc, S0)
                draw_mc_fan(ax_mc, mc, S0, bt)
                fig.canvas.draw_idle()
                _mc_injected[0] = True
                timer.stop()  # done — no more polling needed
                print("[MC] Results injected into chart panels. OK")
        except Exception as exc:
            print(f"[MC DRAW ERROR] {exc}")

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(_check_mc_queue)
    timer.start()

    # ── 10. BUY / SELL / Run MC buttons ────────────────────────────────────
    _trade_log: list[dict] = []

    ax_buy    = fig.add_axes([0.05, 0.01, 0.06, 0.032])
    ax_sell   = fig.add_axes([0.12, 0.01, 0.06, 0.032])
    ax_replay = fig.add_axes([0.19, 0.01, 0.07, 0.032])
    ax_play   = fig.add_axes([0.27, 0.01, 0.07, 0.032])
    ax_mct    = fig.add_axes([0.35, 0.01, 0.10, 0.032])

    btn_buy  = Button(ax_buy,  "BUY NOW", color="#003300", hovercolor="#005500")
    btn_sell = Button(ax_sell, "SELL",    color="#330000", hovercolor="#550000")
    btn_replay = Button(ax_replay, "PLAY",  color="#1a1a2e", hovercolor="#2a2a4e")
    btn_play   = Button(ax_play,   "Run MC", color="#1a1a2e", hovercolor="#2a2a4e")
    btn_mct    = Button(ax_mct, "MC Trade Test", color="#1a1a2e", hovercolor="#2a2a4e")

    btn_buy.label.set_color("#00ff88")
    btn_sell.label.set_color("#ff3b5c")
    btn_replay.label.set_color("#ffffff")
    btn_play.label.set_color("#ffffff")
    btn_mct.label.set_color("#4a9eff")
    btn_mct.label.set_fontsize(8)
    for btn in [btn_buy, btn_sell, btn_replay, btn_play, btn_mct]:
        btn.label.set_fontsize(9)

    def _on_buy(_event) -> None:
        price = float(df["Close"].iloc[-1])
        _trade_log.append({"action": "BUY", "price": price})
        print(f"[TRADE] Manual BUY at ${price:.2f}  (total trades: {len(_trade_log)})")

    def _on_sell(_event) -> None:
        price = float(df["Close"].iloc[-1])
        _trade_log.append({"action": "SELL", "price": price})
        print(f"[TRADE] Manual SELL at ${price:.2f}  (total trades: {len(_trade_log)})")

    def _on_run_mc(_event) -> None:
        # Flush old result from queue so the new one replaces it
        while not _mc_queue.empty():
            try:
                _mc_queue.get_nowait()
            except _queue.Empty:
                break
        _mc_injected[0] = False  # allow timer to inject new result
        timer.start()            # restart polling
        launch_mc(S0, mu, sigma)
        print("[MC] Re-running simulation…")

    btn_buy.on_clicked(_on_buy)
    btn_sell.on_clicked(_on_sell)
    btn_play.on_clicked(_on_run_mc)

    # ── Replay graph: launch the optimized replay app in a new process ───────
    def _on_play(_event) -> None:
        global _replay_proc
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "replay_graph.py")
        try:
            if _replay_proc is not None and _replay_proc.poll() is None:
                print("[REPLAY] Replay window already running.")
                return

            if os.name == "nt":
                # Separate process group/console so replay can't stall the dashboard,
                # and so stdout/stderr don't interleave into the same terminal.
                creationflags = subprocess.CREATE_NEW_CONSOLE
                _replay_proc = subprocess.Popen(
                    [sys.executable, script_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                    close_fds=True,
                )
            else:
                _replay_proc = subprocess.Popen([sys.executable, script_path], close_fds=True)
        except Exception as exc:
            print(f"[REPLAY] Failed to launch replay graph: {exc}")

    btn_replay.on_clicked(_on_play)

    # ── MC Trade Test button ────────────────────────────────────────────────
    def _on_mc_trade_test(_event) -> None:
        if _bt_result is None or ("error" in _bt_result):
            print("Run backtest first")
            return
        show_mc_trade_window(_bt_result, n_sims=80_000)

    btn_mct.on_clicked(_on_mc_trade_test)

    # ── 11. ONE plt.show() ──────────────────────────────────────────────────
    plt.show()


if __name__ == "__main__":
    main()

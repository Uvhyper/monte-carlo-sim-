"""
Tesla Quantitative Analysis
Fetches 5 years of TSLA data, computes technical indicators,
runs Monte Carlo price simulation — opens a SEPARATE Monte Carlo window.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from scipy.stats import t


# ---------------------------------------------------------------------------
TICKER     = "TSLA"
PERIOD     = "5y"
MC_SIMS    = 40_000   # 40k sims — robust stats, half the VRAM of 85k
MC_HORIZON = 252
MC_SHOW    = 350      # Show 350 rendered paths
MC_CHUNK   = 10_000  # Process in 10k-sim chunks to avoid a single 85MB GPU allocation


# ---------------------------------------------------------------------------
def fetch_data(ticker=TICKER, period=PERIOD):
    print(f"[DATA] Fetching {period} of {ticker} data from yfinance...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    
    # Robust date display for logging
    if len(df) > 0:
        start_date = str(df.index[0])[:10]
        end_date = str(df.index[-1])[:10]
        print(f"[DATA] {len(df)} bars loaded ({start_date} -> {end_date})")

    # Technical indicators
    df.ta.sma(length=20,  append=True)
    df.ta.sma(length=50,  append=True)
    df.ta.rsi(length=14,  append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.adx(length=14, append=True)

    # Normalised renames
    rename = {
        "SMA_20":         "SMA 20",
        "SMA_50":         "SMA 50",
        "RSI_14":         "RSI 14",
        "MACD_12_26_9":   "MACD",
        "MACDs_12_26_9":  "MACD_signal",
        "MACDh_12_26_9":  "MACD_hist"
    }
    df.rename(columns=rename, inplace=True)
    
    # Robust BBands rename to handle pandas-ta suffix variations over different versions
    for c in list(df.columns):
        if str(c).startswith("BBL_20"): df.rename(columns={c: "BB Lower"}, inplace=True)
        elif str(c).startswith("BBU_20"): df.rename(columns={c: "BB Upper"}, inplace=True)
        elif str(c).startswith("BBM_20"): df.rename(columns={c: "BB Mid"}, inplace=True)
        elif str(c).startswith("BBB_20"): df.rename(columns={c: "BB Bandwidth"}, inplace=True)
        elif str(c).startswith("BBP_20"): df.rename(columns={c: "BB Percent"}, inplace=True)
    df.dropna(inplace=True)
    return df


def compute_metrics(df):
    close = df["Close"]
    daily = close.pct_change().dropna()
    
    years = len(close) / 252
    # Compound annualised return
    ann_ret = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
    ann_vol = daily.std() * np.sqrt(252)
    
    cum = (1 + daily).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    # Sharpe Ratio with 5.25% risk-free rate
    rf_daily = 0.0525 / 252
    excess_daily = daily - rf_daily
    sharpe = (excess_daily.mean() / daily.std()) * np.sqrt(252) if daily.std() != 0 else 0
    
    # Sortino Ratio (downside only)
    downside = daily[daily < 0]
    down_std = downside.std() * np.sqrt(252)
    sortino = (excess_daily.mean() * 252) / down_std if len(downside) > 0 and down_std != 0 else 0
    
    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    # VaR and CVaR
    var_95 = np.percentile(daily, 5)
    var_99 = np.percentile(daily, 1)
    cvar_95 = daily[daily <= var_95].mean()
    cvar_99 = daily[daily <= var_99].mean()
    
    # Kelly Criterion (cap at 25%)
    wins = daily[daily > 0]
    losses = daily[daily <= 0]
    win_rate = len(wins) / len(daily) if len(daily) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    
    if avg_loss > 0 and avg_win > 0:
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    else:
        kelly = 0
    kelly = min(max(kelly, 0), 0.25)
    
    # Skewness and Kurtosis
    skewness = daily.skew()
    kurtosis = daily.kurtosis()
    
    # Volatility Regime
    vol_20d = daily.rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_252d = daily.rolling(252).std().iloc[-1] * np.sqrt(252)
    if pd.isna(vol_252d): vol_252d = ann_vol
    
    if vol_20d > vol_252d * 1.3:
        regime = "HIGH"
    elif vol_20d < vol_252d * 0.7:
        regime = "LOW"
    else:
        regime = "NORMAL"

    results: dict[str, str] = {
        "Annualised Return": f"{ann_ret * 100:.2f}%",
        "Annualised Volatility": f"{ann_vol * 100:.2f}%",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Calmar Ratio": f"{calmar:.3f}",
        "VaR 95%": f"{var_95 * 100:.2f}%",
        "VaR 99%": f"{var_99 * 100:.2f}%",
        "CVaR 95%": f"{cvar_95 * 100:.2f}%",
        "CVaR 99%": f"{cvar_99 * 100:.2f}%",
        "Kelly Criterion": f"{kelly * 100:.2f}%",
        "Skewness": f"{skewness:.2f}",
        "Kurtosis": f"{kurtosis:.2f}",
        "Volatility Regime": regime
    }
    return results


def monte_carlo(df, n_sims=MC_SIMS, horizon=MC_HORIZON, seed=42):
    print(f"[MC] Running {n_sims:,} simulations over {horizon} trading days...")
    # ✅ FIX: Handle MultiIndex squeeze
    close = df["Close"].values
    while close.ndim > 1: close = close.squeeze()
    
    log_ret = np.diff(np.log(close))
    
    sigma_full = log_ret.std()
    
    # 1. HMM Regime Detection
    try:
        if len(log_ret) > 100:
            hmm_model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=seed)
            hmm_model.fit(log_ret.reshape(-1, 1))
            hidden_state = hmm_model.predict(log_ret.reshape(-1, 1))[-1]
            active_regime = f"HMM State {hidden_state}"
        else:
            active_regime = "UNKNOWN (Not enough data)"
    except Exception as e:
        print(f"[MC] HMM fitting failed: {e}")
        active_regime = "FALLBACK"
        
    # 2. GARCH Volatility Modeling
    try:
        # arch_model expects scaled returns for better convergence
        garch = arch_model(log_ret * 100, vol='GARCH', p=1, q=1)
        res = garch.fit(disp='off')
        forecast = res.forecast(horizon=1)
        # Re-scale variance back to decimal
        sigma = np.sqrt(forecast.variance.values[-1, :][0]) / 100.0
    except Exception as e:
        print(f"[MC] GARCH fitting failed: {e}")
        sigma = log_ret[-30:].std() if len(log_ret) >= 30 else log_ret.std()
        
    # 3. Fat-Tails Student-t Calibration
    try:
        df_est, loc_est, scale_est = t.fit(log_ret)
        dt_df = max(2.1, df_est) # Ensure variance exists
    except Exception as e:
        print(f"[MC] SciPy t-fit failed: {e}")
        dt_df = 5.0

    print(f"[MC] Regime: {active_regime} | GARCH Vol: {sigma*np.sqrt(252)*100:.2f}% | t-dist df: {dt_df:.2f}")
        
    # Itô-corrected GBM drift
    mu = log_ret.mean() - 0.5 * (sigma ** 2)
    last = float(close[-1])

    mu    = np.float32(mu)
    sigma = np.float32(sigma)

    # CPU/GPU optimization for Monte Carlo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MC] Using {device.type.upper()} for high-performance computations...")
    torch.manual_seed(seed)

    # Chunked Monte Carlo — prevents a single giant tensor allocation
    # Accumulate running stats and keep only display paths.
    chunk_size = min(MC_CHUNK, n_sims)
    n_chunks   = (n_sims + chunk_size - 1) // chunk_size

    # Accumulators (on CPU to save VRAM between chunks)
    running_sum   = np.zeros(MC_HORIZON, dtype=np.float64)  # for avg_path
    running_sum2  = np.zeros(MC_HORIZON, dtype=np.float64)  # for variance
    all_ends      = []                                        # terminal prices
    paths_show    = []                                        # sampled display paths
    paths_show_budget = MC_SHOW

    mu_t   = torch.tensor(float(mu),    device=device)
    sig_t  = torch.tensor(float(sigma), device=device)
    last_t = torch.tensor(last,         device=device)
    adj_var = float(dt_df / (dt_df - 2.0))
    
    from torch.distributions.studentT import StudentT
    dist = StudentT(df=torch.tensor([float(dt_df)], device=device))

    with torch.no_grad():
        for ci in range(n_chunks):
            actual_sims = min(chunk_size, n_sims - ci * chunk_size)
            if actual_sims <= 0:
                break

            rand = dist.sample((MC_HORIZON, actual_sims)).squeeze(-1)
            rand = rand / np.sqrt(adj_var)
            rand = torch.clamp(rand, -4.0, 4.0)

            step  = torch.exp(mu_t + sig_t * rand)            # (H, chunk)
            paths = last_t * torch.cumprod(step, dim=0)       # (H, chunk)

            # Accumulate running stats (on CPU)
            paths_np = paths.cpu().numpy()
            running_sum  += paths_np.sum(axis=1)
            running_sum2 += (paths_np ** 2).sum(axis=1)
            all_ends.append(paths_np[-1, :])

            # Sample display paths proportionally from each chunk
            n_show_this = max(1, paths_show_budget * actual_sims // n_sims)
            idxs        = np.random.choice(actual_sims, min(n_show_this, actual_sims), replace=False)
            paths_show.append(paths_np[:, idxs])

            # ✅ VRAM released immediately after each chunk
            del rand, step, paths
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Assemble final stats
    all_ends_np  = np.concatenate(all_ends)
    paths_show   = np.concatenate(paths_show, axis=1)[:, :MC_SHOW]
    avg_path     = (running_sum  / n_sims).astype(np.float32)
    med_path     = np.median(np.concatenate([p for p in [np.array([e]) for e in all_ends]], axis=0)) * np.ones(MC_HORIZON, dtype=np.float32)  # scalar fallback
    
    # Build percentile paths from accumulated chunk data (recompute with lightweight re-run)
    # Use only MC_SHOW samples (already stored) for percentile path bands
    p5_path  = np.percentile(paths_show, 5,  axis=1).astype(np.float32)
    p95_path = np.percentile(paths_show, 95, axis=1).astype(np.float32)
    p1_path  = np.percentile(paths_show, 1,  axis=1).astype(np.float32)
    p99_path = np.percentile(paths_show, 99, axis=1).astype(np.float32)

    p_mean   = float(all_ends_np.mean())
    p_median = float(np.median(all_ends_np))
    p1, p5, p95, p99 = p1_path[-1], p5_path[-1], p95_path[-1], p99_path[-1]
    med_path = np.full(MC_HORIZON, p_median, dtype=np.float32)

    print(f"[MC] ${last:.2f} now | Mean ${p_mean:.2f} | Median ${p_median:.2f} | 1-5-95-99%: ${p1:.2f}, ${p5:.2f}, ${p95:.2f}, ${p99:.2f}")

    return dict(
        paths=paths_show, last_price=last,
        p_mean=p_mean, p_median=p_median,
        perc_1=p1, perc_5=p5, perc_95=p95, perc_99=p99,
        p5_path=p5_path,
        p95_path=p95_path,
        p1_path=p1_path,
        p99_path=p99_path,
        avg_path=avg_path,
        med_path=med_path,
        horizon=MC_HORIZON, drift=float(mu), sigma=float(sigma)
    )



def show_monte_carlo_window(mc_result):
    """Open a standalone Monte Carlo window (matches original screenshots)."""
    paths      = mc_result["paths"]
    last       = mc_result["last_price"]
    p_mean = mc_result["p_mean"]
    horizon    = mc_result["horizon"]
    fwd        = np.arange(horizon)

    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Use purely Object-Oriented Matplotlib. NEVER use `plt.subplots()` 
    # to avoid the global state machine crashing inside Tkinter event loops.
    fig = Figure(figsize=(10, 6), facecolor="#0d1117", dpi=100)
    ax = fig.add_subplot(111, facecolor="#0d1117")
    
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.set_xlabel("Days Future", color="#c9d1d9")
    ax.set_ylabel("Price (USD)", color="#c9d1d9")
    ax.set_title(f"Monte Carlo Simulation of TSLA Stock Price "
                 f"({MC_SIMS:,} paths computed, {MC_SHOW:,} shown)", color="#c9d1d9")

    from matplotlib.collections import LineCollection
    import pandas as pd

    # Draw sample paths (deep, vibrant colours)
    cmap = matplotlib.colormaps['nipy_spectral']
    
    # Extremely fast GPU/CPU PyTorch 1D convolution smoothing (bypasses Pandas DataFrame lag)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_selected = paths
    
    with torch.no_grad(): # Keep it purely computationally, no learning memory overhead
        y_selected_t = torch.tensor(y_selected, dtype=torch.float32, device=device).t().unsqueeze(1)
        # Replicate padding correctly averages edges instead of artificially crashing them down tracking a 0 placeholder
        y_padded = torch.nn.functional.pad(y_selected_t, (3, 3), mode='replicate')
        kernel = torch.ones(1, 1, 7, dtype=torch.float32, device=device) / 7.0
        y_smooth_t = torch.nn.functional.conv1d(y_padded, kernel)
        y_smooth = y_smooth_t.squeeze(1).t().cpu().numpy()
        
        # Release the VRAM immediately after use
        del y_selected_t, y_padded, kernel, y_smooth_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Ultra-fast vectorized plot segment creation
    n_show = y_selected.shape[1]
    
    # DOWNSAMPLING: Only plot every 4th day to slice the rendering workload by 75%
    ds_factor = max(1, horizon // 60) # targeting ~60 render points per simulation
    fwd_ds = fwd[::ds_factor]
    y_smooth_ds = y_smooth[::ds_factor, :]
    
    segs = np.stack([np.broadcast_to(fwd_ds[:, None], (len(fwd_ds), n_show)), y_smooth_ds], axis=-1)
    segments = segs.transpose(1, 0, 2)
    colors = cmap(np.linspace(0, 1, n_show))
        
    # Optimise plotting using LineCollection instead of 1000 individual plot calls
    lc = LineCollection(list(segments), colors=colors, alpha=0.35, linewidths=0.8)
    ax.add_collection(lc)

    # Average path
    avg = mc_result.get("avg_path", paths.mean(axis=1))
    ax.plot(fwd, avg, color="white", lw=3, label="Average Path", zorder=5)
    ax.axhline(last, color="#8b949e", lw=1, ls="--")

    pct = (p_mean / last - 1) * 100
    dir_sym = "+" if pct >= 0 else ""
    ax.annotate(f"  {dir_sym}{pct:.1f}%  ${p_mean:.2f} (252d)",
                xy=(horizon - 1, p_mean), color="#3fb950" if pct >= 0 else "#f85149", fontsize=9)

    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    ax.yaxis.set_tick_params(labelcolor="#c9d1d9")
    ax.xaxis.set_tick_params(labelcolor="#c9d1d9")
    
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"[MC GRAPH WARNING] Layout adjusting error: {e}")
        
    # Construct an explicitly isolated native Tkinter window on the main thread.
    # We do NOT use `tk.Tk()` here to avoid a "Double Root" Tcl exception, as
    # Matplotlib already generated a default Tk event root in `replay_graph.py`.
    mc_window = tk.Toplevel()
    mc_window.title("Monte Carlo Simulation - TSLA")
    mc_window.geometry("800x500")
    mc_window.configure(bg="#0d1117")

    canvas = FigureCanvasTkAgg(fig, master=mc_window)
    canvas.draw_idle()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    return mc_window


def run_analysis():
    try:
        df = fetch_data()
    except Exception as e:
        print(f"[ERROR] Data fetch failed: {e}")
        return None

    metrics = compute_metrics(df)
    print("\n" + "="*40)
    print(f"  QUANT PERFORMANCE METRICS: {TICKER}")
    print("="*40)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("="*40 + "\n")

    mc_result = monte_carlo(df)
    mc_fig = show_monte_carlo_window(mc_result)
    return df, mc_result, mc_fig

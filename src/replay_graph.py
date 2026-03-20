"""
TSLA Replay Graph — fast, no-lag version using set_data() pattern.
- Pre-creates all line artist objects once
- Uses set_data() each frame (no ax.plot() in the animation loop)
- blit=True so ONLY changed artists are redrawn (kills CPU spike)
- Starts at bar 1 and plays automatically
- BUY NOW / SELL / Pause buttons at bottom-right
- MASTER AI panel with D1-D5 forecast colour bars
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

# ✅ FIX: Aggressive path simplification to reduce render cost
plt.rcParams["path.simplify"]           = True
plt.rcParams["path.simplify_threshold"] = 1.0
plt.rcParams["agg.path.chunksize"]      = 10000
plt.rcParams["toolbar"]                 = "None"


# ---------------------------------------------------------------------------
def _col(df, *names):
    for n in names:
        if n in df.columns:
            val = df[n].values
            while val.ndim > 1:
                val = val.squeeze()
            return val
    return None


def compute_signals(df):
    n      = len(df)
    votes  = np.zeros(n, dtype=int)
    close  = df["Close"].values
    while close.ndim > 1: close = close.squeeze()

    _macd   = _col(df, "MACD", "MACD_12_26_9")
    _macd_s = _col(df, "MACD_signal", "MACDs_12_26_9")
    _rsi    = _col(df, "RSI 14", "RSI_14", "RSI")
    _s20    = _col(df, "SMA 20", "SMA_20")
    _s50    = _col(df, "SMA 50", "SMA_50")
    _bbl    = _col(df, "BB Lower")
    _bbu    = _col(df, "BB Upper")
    _adx    = _col(df, "ADX_14")

    macd   = _macd   if _macd   is not None else np.zeros(n)
    macd_s = _macd_s if _macd_s is not None else np.zeros(n)
    rsi    = _rsi    if _rsi    is not None else np.full(n, 50.0)
    s20    = _s20    if _s20    is not None else close.copy()
    s50    = _s50    if _s50    is not None else close.copy()
    bbl    = _bbl    if _bbl    is not None else np.zeros(n)
    bbu    = _bbu    if _bbu    is not None else np.zeros(n)
    adx    = _adx    if _adx    is not None else np.full(n, 20.0)

    # ---------------------------------------------------------
    # CPU VECTORIZATION: Instant 5-year analysis with NumPy. 
    # Distributes load perfectly to CPU, ignoring GPU completely.
    # ---------------------------------------------------------
    
    # 1. MACD Crossover
    macd_bull = (macd[:-1] < macd_s[:-1]) & (macd[1:] >= macd_s[1:])
    macd_bear = (macd[:-1] > macd_s[:-1]) & (macd[1:] <= macd_s[1:])
    votes[1:][macd_bull] += 2
    votes[1:][macd_bear] -= 2

    # 2. SMA Golden/Death cross
    valid_sma = ~np.isnan(s20[1:]) & ~np.isnan(s50[1:]) & ~np.isnan(s20[:-1]) & ~np.isnan(s50[:-1])
    sma_bull = valid_sma & (s20[:-1] < s50[:-1]) & (s20[1:] >= s50[1:])
    sma_bear = valid_sma & (s20[:-1] > s50[:-1]) & (s20[1:] <= s50[1:])
    votes[1:][sma_bull] += 1
    votes[1:][sma_bear] -= 1

    # 3. BB Mean reversion
    in_range = adx[1:] < 25
    bb_bull = in_range & (close[1:] < bbl[1:])
    bb_bear = in_range & (close[1:] > bbu[1:])
    votes[1:][bb_bull] += 1
    votes[1:][bb_bear] -= 1

    # 4. RSI Extremes
    rsi_bull = rsi[1:] < 32
    rsi_bear = rsi[1:] > 68
    votes[1:][rsi_bull] += 1
    votes[1:][rsi_bear] -= 1

    # 5. ADX Gate
    thresholds = np.where(adx >= 20, 2, 3)
    
    sig_arr = np.empty(n, dtype=object)
    sig_arr[:] = ""
    sig_arr[votes >= thresholds] = "BUY"
    sig_arr[votes <= -thresholds] = "SELL"

    return sig_arr.tolist()


# ---------------------------------------------------------------------------
DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"
BLUE   = "#58a6ff"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"


class InteractiveReplayGraph:

    def __init__(self, df, ticker="TSLA", mc_result=None, master_result=None):
        self.pending_mc = None
        self.pending_master = None
        self.df            = df.reset_index()
        self.ticker        = ticker
        self.mc_result     = mc_result
        self.master_result = master_result
        self.n             = len(self.df)
        
        # ✅ FIX: Pre-flatten all column arrays at startup
        raw = self.df["Close"].values
        while raw.ndim > 1: raw = raw.squeeze()
        self.close = raw.astype(float)
        
        self._c_sma20 = _col(self.df, "SMA 20", "SMA_20")
        self._c_sma50 = _col(self.df, "SMA 50", "SMA_50")
        self._c_bbu   = _col(self.df, "BB Upper")
        self._c_bbl   = _col(self.df, "BB Lower")
        self._c_rsi   = _col(self.df, "RSI 14", "RSI_14", "RSI")
        
        self.xs = np.arange(self.n, dtype=float)

        self.signals  = compute_signals(df)
        buy_n  = self.signals.count("BUY")
        sell_n = self.signals.count("SELL")
        print(f"  -> {buy_n} BUY signals, {sell_n} SELL signals found across history.")
        self.buy_idx  = np.array([i for i, s in enumerate(self.signals) if s == "BUY"],  dtype=int)
        self.sell_idx = np.array([i for i, s in enumerate(self.signals) if s == "SELL"], dtype=int)

        self.trades      = []
        self._trade_history_bars_buy = np.array([], dtype=int)
        self._trade_history_bars_sell = np.array([], dtype=int)
        self.current_bar = 1
        self.is_playing  = True
        self.interval_ms = 50  # Target ~20 FPS for rendering headroom
        
        # Stability state
        import time
        self._needs_redraw = False
        self._is_drag_paused = False
        self._drag_pause_id  = None
        
        # Blitting core
        self.background = None
        self._first_draw = True
        self._last_frame_time = time.time()

        # ---- Figure --------------------------------------------------------
        import matplotlib.gridspec as gridspec
        has_master = master_result is not None
        heights = [4, 1.4, 1.8, 2.0] if has_master else [4, 1.4, 1.8]
        n_rows  = len(heights)

        # ✅ FIX: Lower DPI (80) reduces pixel buffer pressure
        self.fig = plt.figure(figsize=(15, 12), facecolor=DARK, dpi=80)
        
        # GridSpec for charts (left) and metrics sidebar (right)
        gs_main = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[4, 1], 
                                   wspace=0.03, left=0.05, right=0.99, top=0.94, bottom=0.08)
        gs_left = gridspec.GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs_main[0], 
                                                  height_ratios=heights, hspace=0.06)

        try:
            self.fig.canvas.manager.set_window_title("TSLA Replay Graph")
        except Exception:
            pass

        self.ax_price  = self.fig.add_subplot(gs_left[0])
        self.ax_rsi    = self.fig.add_subplot(gs_left[1])
        self.ax_mc     = self.fig.add_subplot(gs_left[2])
        self.ax_master = self.fig.add_subplot(gs_left[3]) if has_master else None

        self.ax_sidebar = self.fig.add_subplot(gs_main[1])
        self.ax_sidebar.set_facecolor(PANEL)
        self.ax_sidebar.axis("off")

        axes = [self.ax_price, self.ax_rsi, self.ax_mc]
        if has_master:
            axes.append(self.ax_master)

        for ax in axes:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        self._title = self.fig.text(
            0.5, 0.965,
            f"{ticker}  \u00b7  bar 1 / {self.n}",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=BORDER),
        )

        self._build_axes()
        self._build_lines()
        self._setup_buttons()
        self._draw_static_overlays()
        self._update(1)

    def inject_results(self, mc_result, master_result):
        self.mc_result     = mc_result
        self.master_result = master_result
        self._draw_mc_panel()
        self._draw_metrics_panel()
        if master_result and self.ax_master:
            self._draw_master_panel()
        self._needs_redraw = True

    # -----------------------------------------------------------------------
    def _build_axes(self):
        c = self.close
        self.ax_price.set_ylabel("Price (USD)", color=MUTED, fontsize=8)
        self.ax_price.set_xlim(0, self.n)
        pad = max((c.max() - c.min()) * 0.06, 5)
        self.ax_price.set_ylim(c.min() - pad, c.max() + pad)
        self.ax_price.xaxis.set_visible(False)
        self.ax_price.set_title(
            f"{self.ticker}  ·  Price & Technical Indicators"
            "   (^ = Auto BUY  |  v = Auto SELL)",
            color=TEXT, fontsize=9, pad=4,
        )
        self.ax_rsi.set_ylabel("RSI 14", color=MUTED, fontsize=8)
        self.ax_rsi.set_xlim(0, self.n)
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(70, color=RED,   lw=0.8, ls="--", alpha=0.6)
        self.ax_rsi.axhline(30, color=GREEN, lw=0.8, ls="--", alpha=0.6)
        self.ax_rsi.xaxis.set_visible(False)
        self.ax_mc.set_ylabel("USD", color=MUTED, fontsize=8)
        self.ax_mc.tick_params(axis="x", colors=MUTED, labelsize=7)

    # -----------------------------------------------------------------------
    def _build_lines(self):
        """Create all artist objects once. set_data() updates them — no new plots."""
        ax = self.ax_price

        # Static ghost lines (drawn once, never updated)
        ax.plot(self.xs, self.close, color=BORDER, lw=1, alpha=0.35, zorder=1)
        if self._c_sma20 is not None: ax.plot(self.xs, self._c_sma20, color=BLUE, lw=0.5, alpha=0.15, zorder=1)
        if self._c_sma50 is not None: ax.plot(self.xs, self._c_sma50, color=ORANGE, lw=0.5, alpha=0.15, zorder=1)
        if self._c_bbu is not None: ax.plot(self.xs, self._c_bbu, color=MUTED, lw=0.5, alpha=0.15, zorder=1)
        if self._c_bbl is not None: ax.plot(self.xs, self._c_bbl, color=MUTED, lw=0.5, alpha=0.15, zorder=1)

        # ✅ Animated live lines — these are the ONLY things blit redraws
        self._ln_close,  = ax.plot([], [], color=BLUE,   lw=1.5, zorder=3, label="Close")
        self._ln_sma20,  = ax.plot([], [], color=BLUE,   lw=1,   zorder=2, alpha=0.85,
                                   ls="--", label="SMA 20")
        self._ln_sma50,  = ax.plot([], [], color=ORANGE, lw=1,   zorder=2, alpha=0.85,
                                   ls="--", label="SMA 50")
        self._ln_bbu,    = ax.plot([], [], color=MUTED,  lw=0.7, zorder=2, alpha=0.6,
                                   label="BB Upper")
        self._ln_bbl,    = ax.plot([], [], color=MUTED,  lw=0.7, zorder=2, alpha=0.6,
                                   label="BB Lower")

        self._sc_buy   = ax.scatter([], [], marker="^", s=70, color=GREEN,    zorder=5,
                                    edgecolors="white", linewidths=0.5, label="Auto BUY")
        self._sc_sell  = ax.scatter([], [], marker="v", s=70, color=RED,      zorder=5,
                                    edgecolors="white", linewidths=0.5, label="Auto SELL")
        self._sc_mbuy  = ax.scatter([], [], marker="D", s=80, color="#ffa657",zorder=6,
                                    edgecolors="white", linewidths=0.5, label="Manual BUY")
        self._sc_msell = ax.scatter([], [], marker="X", s=80, color=YELLOW,   zorder=6,
                                    edgecolors="white", linewidths=0.5, label="Manual SELL")

        ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
                  labelcolor=TEXT, ncol=2)

        # RSI live line
        self._ln_rsi, = self.ax_rsi.plot([], [], color=PURPLE, lw=1, zorder=3,
                                          label="RSI 14")
        if self._c_rsi is not None:
            self.ax_rsi.plot(self.xs, self._c_rsi, color=PURPLE, lw=0.5, alpha=0.15, zorder=1)

    # -----------------------------------------------------------------------
    def _setup_buttons(self):
        ax_buy   = self.fig.add_axes([0.70, 0.025, 0.09, 0.04])
        ax_sell  = self.fig.add_axes([0.80, 0.025, 0.09, 0.04])
        ax_pause = self.fig.add_axes([0.90, 0.025, 0.09, 0.04])

        self.btn_buy   = Button(ax_buy,   "BUY NOW", color="#1a4731", hovercolor=GREEN)
        self.btn_sell  = Button(ax_sell,  "SELL",    color="#4a1a1a", hovercolor=RED)
        self.btn_pause = Button(ax_pause, "Pause",   color=PANEL,     hovercolor=BORDER)

        for btn in [self.btn_buy, self.btn_sell, self.btn_pause]:
            btn.label.set_color(TEXT)
            btn.label.set_fontsize(9)

        self.btn_buy.on_clicked(self._manual_buy)
        self.btn_sell.on_clicked(self._manual_sell)
        self.btn_pause.on_clicked(self._toggle_play)

    # -----------------------------------------------------------------------
    def _draw_static_overlays(self):
        """MC panel + MASTER panel — drawn ONCE, never touched again by blit."""
        self._draw_mc_panel()
        if self.master_result and self.ax_master is not None:
            self._draw_master_panel()
        self._draw_metrics_panel()

    # -----------------------------------------------------------------------
    def _draw_metrics_panel(self):
        ax = self.ax_sidebar
        ax.set_facecolor(PANEL)
        ax.axis("off")
        
        if not self.mc_result:
            return
            
        ax.text(0.1, 0.95, "Quant Metrics", color=TEXT, fontsize=12, fontweight="bold", transform=ax.transAxes)
        y = 0.88
        dy = 0.04
        
        for k, v in [
            ("Mean Price", f"${self.mc_result.get('p_mean', 0):.2f}"),
            ("Median Price", f"${self.mc_result.get('p_median', 0):.2f}"),
            ("1st Percentile", f"${self.mc_result.get('perc_1', 0):.2f}"),
            ("5th Percentile", f"${self.mc_result.get('perc_5', 0):.2f}"),
            ("95th Percentile", f"${self.mc_result.get('perc_95', 0):.2f}"),
            ("99th Percentile", f"${self.mc_result.get('perc_99', 0):.2f}"),
            ("Drift", f"{self.mc_result.get('drift', 0):.6f}"),
            ("Sigma", f"{self.mc_result.get('sigma', 0):.6f}"),
        ]:
            ax.text(0.1, y, k, color=MUTED, fontsize=9, transform=ax.transAxes)
            ax.text(0.95, y, v, color=TEXT, fontsize=9, ha="right", transform=ax.transAxes)
            y -= dy

    # -----------------------------------------------------------------------
    def _draw_mc_panel(self):
        ax = self.ax_mc
        if not self.mc_result:
            ax.text(0.5, 0.5, "Monte Carlo not available",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes)
            return

        paths  = self.mc_result["paths"]
        last   = self.mc_result["last_price"]
        p_mean = self.mc_result.get("p_mean", paths[:, -1].mean())
        h      = paths.shape[0]
        xs     = np.arange(h)

        # ✅ FIX: Use precomputed percentiles from mc_result
        p5  = self.mc_result["p5_path"]
        p95 = self.mc_result["p95_path"]
        avg = self.mc_result["avg_path"]

        ax.fill_between(xs, p5, p95, alpha=0.25, color=BLUE, label="5-95% confidence")
        ax.plot(xs, avg, color=ORANGE, lw=1.5, label="Mean predicted path")
        ax.axhline(last, color=MUTED, lw=0.8, ls="--", label=f"Current ${last:.2f}")

        pct = (p_mean / last - 1) * 100
        clr = GREEN if pct >= 0 else RED
        sym = "+" if pct >= 0 else ""
        ax.annotate(
            f"  {'▲' if pct >= 0 else '▼'} ${p_mean:.2f} ({h}d)",
            xy=(h - 1, p_mean), color=clr, fontsize=8, ha="right",
        )

        n_sims = paths.shape[1]
        ax.set_title(
            f"Predicted Price  (Monte Carlo \u2014 {n_sims:,} sims, {h} days)",
            color=TEXT, fontsize=8, pad=4,
        )
        ax.set_xlabel("Days into the Future", color=MUTED, fontsize=7)
        ax.legend(fontsize=7, framealpha=0.3, labelcolor=TEXT, loc="upper left")

    # -----------------------------------------------------------------------
    def _draw_master_panel(self):
        ax  = self.ax_master
        res = self.master_result
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")

        signal     = res.get("signal", "HOLD")
        confidence = res.get("confidence", 0.0)
        pred_ret   = res.get("predicted_return_pct", 0.0)
        forecast   = res.get("forecast_prices", [])

        sig_clr = GREEN if signal == "BUY" else (RED if signal == "SELL" else YELLOW)
        bg_clr  = "#0d2119" if signal == "BUY" else ("#2d0f0e" if signal == "SELL" else "#2d2208")
        arrow   = "\u2191" if signal == "BUY" else ("\u2193" if signal == "SELL" else "\u2022")

        ax.text(0.3, 3.75, "\U0001f916  MASTER AI Transformer Prediction",
                fontsize=10, color=TEXT, fontweight="bold")

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.3, 1.85), 2.2, 1.6,
            boxstyle="round,pad=0.1", facecolor=bg_clr,
            edgecolor=sig_clr, linewidth=2,
        ))
        ax.text(1.4, 3.1, arrow,  ha="center", fontsize=16, color=sig_clr, fontweight="bold")
        ax.text(1.4, 2.4, signal, ha="center", fontsize=13, color=sig_clr, fontweight="bold")

        ax.text(3.0, 3.5, f"Confidence: {confidence:.0%}", fontsize=9, color=TEXT)
        ax.add_patch(mpatches.Rectangle((3.0, 3.15), 5.5, 0.22, facecolor=BORDER, linewidth=0))
        ax.add_patch(mpatches.Rectangle((3.0, 3.15), 5.5 * min(confidence, 1.0), 0.22,
                                         facecolor=sig_clr, alpha=0.8, linewidth=0))

        ret_clr = GREEN if pred_ret >= 0 else RED
        sign    = "+" if pred_ret >= 0 else ""
        ax.text(3.0, 2.7, f"Predicted 1-day return: {sign}{pred_ret:.4f}%",
                fontsize=9, color=ret_clr)

        ax.text(3.0, 2.3, "5-Day Forecast:", fontsize=8, color=TEXT)
        if len(forecast) >= 2:
            base  = forecast[0]
            bar_w = 0.9
            for k, price in enumerate(forecast[:5]):
                x   = 3.0 + k * (bar_w + 0.15)
                pct = (price - base) / max(abs(base), 1e-6)
                bh  = max(abs(pct) * 15, 0.06)
                clr = GREEN if price >= base else RED
                ax.add_patch(mpatches.Rectangle(
                    (x, 1.05), bar_w, bh, facecolor=clr, alpha=0.8, linewidth=0))
                ax.text(x + bar_w / 2, 0.88, f"D{k+1}",
                        ha="center", fontsize=7, color=MUTED)
                base = price

    # -----------------------------------------------------------------------
    def _update(self, bar_idx):
        i = max(1, min(int(bar_idx), self.n))

        self._ln_close.set_data(self.xs[:i], self.close[:i])

        if self._c_sma20 is not None: self._ln_sma20.set_data(self.xs[:i], self._c_sma20[:i])
        if self._c_sma50 is not None: self._ln_sma50.set_data(self.xs[:i], self._c_sma50[:i])
        if self._c_bbu is not None: self._ln_bbu.set_data(self.xs[:i], self._c_bbu[:i])
        if self._c_bbl is not None: self._ln_bbl.set_data(self.xs[:i], self._c_bbl[:i])

        if self._c_rsi is not None:
            self._ln_rsi.set_data(self.xs[:i], self._c_rsi[:i])

        # Optimize auto signal searches using searchsorted (O(log N) instead of O(N))
        idx_b = np.searchsorted(self.buy_idx, i)
        idx_s = np.searchsorted(self.sell_idx, i)
        mask_b = self.buy_idx[:idx_b]
        mask_s = self.sell_idx[:idx_s]

        # Only call set_offsets if the count of points to draw has changed
        if idx_b != getattr(self, '_last_idx_b', -1) or i == 1:
            self._sc_buy.set_offsets(
                np.c_[mask_b, self.close[mask_b]] if idx_b > 0 else np.empty((0, 2)))
            self._last_idx_b = idx_b

        if idx_s != getattr(self, '_last_idx_s', -1) or i == 1:
            self._sc_sell.set_offsets(
                np.c_[mask_s, self.close[mask_s]] if idx_s > 0 else np.empty((0, 2)))
            self._last_idx_s = idx_s

        # Optimize manual trade lookups using boolean masks (since they can be unsorted upon playback loops)
        mbx = self._trade_history_bars_buy[self._trade_history_bars_buy < i]
        msx = self._trade_history_bars_sell[self._trade_history_bars_sell < i]

        if len(mbx) != getattr(self, '_last_mbx_len', -1) or i == 1:
            self._sc_mbuy.set_offsets(
                np.c_[mbx, self.close[mbx]] if len(mbx) > 0 else np.empty((0, 2)))
            self._last_mbx_len = len(mbx)

        if len(msx) != getattr(self, '_last_msx_len', -1) or i == 1:
            self._sc_msell.set_offsets(
                np.c_[msx, self.close[msx]] if len(msx) > 0 else np.empty((0, 2)))
            self._last_msx_len = len(msx)

        self._title.set_text(f"{self.ticker}  \u00b7  bar {i} / {self.n}")
        
        # Execute True Blitting instead of massive redraws
        if self._first_draw:
            # First pass: let Matplotlib draw the whole grid once
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self._first_draw = False
        else:
            if self.background is not None:
                self.fig.canvas.restore_region(self.background)
                
            # Redraw only the dynamic lines and scatters
            self.ax_price.draw_artist(self._ln_close)
            self.ax_price.draw_artist(self._ln_sma20)
            self.ax_price.draw_artist(self._ln_sma50)
            self.ax_price.draw_artist(self._ln_bbu)
            self.ax_price.draw_artist(self._ln_bbl)
            self.ax_price.draw_artist(self._sc_buy)
            self.ax_price.draw_artist(self._sc_sell)
            self.ax_price.draw_artist(self._sc_mbuy)
            self.ax_price.draw_artist(self._sc_msell)
            
            # Redraw title
            self.fig.draw_artist(self._title)
            
            # Redraw RSI
            self.ax_rsi.draw_artist(self._ln_rsi)
            
            self.fig.canvas.blit(self.fig.bbox)

    # -----------------------------------------------------------------------
    def _throttled_draw(self):
        # We handle blitting directly in _update, so standard draw requests are 
        # only flagged when doing massive state changes (like appending MC charts)
        self._needs_redraw = True

    def _flush_draw(self):
        if self._needs_redraw:
            self._needs_redraw = False
            self.fig.canvas.draw_idle()
            # Force background reset on next frame so blitting picks up new axes
            self._first_draw = True

    def _start_flush_timer(self):
        if getattr(self, 'pending_mc', None) is not None:
            mc, master = self.pending_mc, self.pending_master
            self.pending_mc = None
            self.pending_master = None
            self.inject_results(mc, master)
            
            try:
                # We return an explicit tk.Toplevel window now from quant_analysis.
                # Since it's Object-Oriented and not Pyplot-driven, there are ZERO race conditions.
                from quant_analysis import show_monte_carlo_window
                self.mc_window = show_monte_carlo_window(mc)
                # Tkinter windows auto-display on instantiation, no `.show()` is necessary.
            except Exception as e:
                print(f"[MAIN THREAD] Could not open MC GUI window: {e}")

        self._flush_draw()
        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.get_tk_widget().after(60, self._start_flush_timer)

    def _on_window_configure(self, event):
        """Pause animation while window is being moved/resized to prevent Watchdog crash."""
        if self._drag_pause_id is not None:
            self.fig.canvas.get_tk_widget().after_cancel(self._drag_pause_id)
        
        self._is_drag_paused = True
        if hasattr(self, '_anim') and self._anim.event_source:
            self._anim.event_source.stop()
            
        # Resume 500ms after movement stops
        self._drag_pause_id = self.fig.canvas.get_tk_widget().after(500, self._resume_after_drag)

    def _resume_after_drag(self):
        self._is_drag_paused = False
        self._drag_pause_id  = None
        self._throttled_draw()

    def _animate(self, _frame):
        try:
            # Check for active Matplotlib UI interactions (pan/zoom)
            active_tool = self.fig.canvas.toolbar.mode if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar else ""
            if active_tool != "":
                # Suspend playback frame advancing while user is using a tool
                return
                
            if not self.is_playing or self._is_drag_paused:
                return
                
            if self.current_bar < self.n:
                self.current_bar += 1
                self._update(self.current_bar)
            else:
                self.is_playing = False
                self.btn_pause.label.set_text("Play")
                # Trigger a full redraw on completion to finalize state
                self._throttled_draw()
                self._flush_draw()
        except Exception as e:
            print(f"[ANIMATE ERROR] {e}")

    def _toggle_play(self, _event=None):
        if not self.is_playing and self.current_bar >= self.n:
            self.current_bar = 1
        self.is_playing = not self.is_playing
        self.btn_pause.label.set_text("Pause" if self.is_playing else "Play")

    def _manual_buy(self, _event=None):
        bar = max(self.current_bar - 1, 0)
        self.trades.append({"action": "BUY", "bar": bar})
        self._trade_history_bars_buy = np.append(self._trade_history_bars_buy, bar)
        self._update(self.current_bar)
        self._throttled_draw()

    def _manual_sell(self, _event=None):
        bar = max(self.current_bar - 1, 0)
        self.trades.append({"action": "SELL", "bar": bar})
        self._trade_history_bars_sell = np.append(self._trade_history_bars_sell, bar)
        self._update(self.current_bar)
        self._throttled_draw()

    def show(self):
        # Bind drag/resize pause logic
        self.fig.canvas.get_tk_widget().winfo_toplevel().bind("<Configure>", self._on_window_configure)
        
        # Tell Tkinter to process events before matplotlib redraws
        plt.rcParams["tk.window_focus"] = False
        
        # Start the flushing timer
        self._start_flush_timer()
        
        # Start our own manual, crash-proof playback loop (replaces FuncAnimation)
        self._playback_loop()
        
        plt.show()

    def _playback_loop(self):
        # Rely natively on Tkinter's event loop pacing
        if plt.fignum_exists(self.fig.number):
            self._animate(None)
            self.fig.canvas.get_tk_widget().after(self.interval_ms, self._playback_loop)


if __name__ == "__main__":
    import os
    import sys

    # Allow `python src/replay_graph.py` to import `src/main.py`
    sys.path.insert(0, os.path.dirname(__file__))

    from main import load_tsla  # reuse the cached safety-first loader

    print("=" * 62)
    print("  TSLA Replay Graph (optimized)")
    print("=" * 62)

    df = load_tsla(period="5y", interval="1d")

    # Indicators expected by replay graph + signal logic inside it
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.dropna(inplace=True)

    app = InteractiveReplayGraph(df, ticker="TSLA")
    app.show()

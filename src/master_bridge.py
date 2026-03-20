"""
MASTER Bridge Module
Lightweight MiniMASTER transformer trained on TSLA data.
Provides buy/sell signal + 5-day return forecast.
Runs on CPU only, safe for regular PCs.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader

TORCH_AVAILABLE = True
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory
        if free_mem > 2_000_000_000:   # need > 2GB VRAM
            DEVICE = "cuda:0"
            torch.backends.cudnn.benchmark = True
            print(f"[MASTER] GPU: {torch.cuda.get_device_name(0)} ({free_mem/1e9:.1f}GB)")
        else:
            DEVICE = "cpu"
            print(f"[MASTER] GPU < 2GB VRAM — using CPU")
    else:
        DEVICE = "cpu"
else:
    DEVICE = "cpu"

SEQ_LEN     = 20
N_FEATURES  = 15
HIDDEN_DIM  = 64    # restored original
N_HEADS     = 4     # restored original
N_LAYERS    = 2     # restored original
PRED_DAYS   = 5
MAX_EPOCHS  = 30
BATCH_SIZE  = 16
MODEL_PATH  = "src/master_model/mini_master_weights.pt"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build a feature matrix of shape (n_bars, N_FEATURES).
    Uses only robust, pandas-derivable features.
    """
    # ✅ FIX: Squeeze all input features to 1D arrays
    def _get_val(col):
        if col not in df.columns: return None
        val = df[col].values.astype(float)
        while val.ndim > 1: val = val.squeeze()
        return val

    close  = _get_val("Close")
    if close is None:
        return np.zeros((len(df), N_FEATURES), dtype=np.float32)
        
    high_val = _get_val("High")   
    high = high_val if high_val is not None else close
    
    low_val = _get_val("Low")    
    low = low_val if low_val is not None else close
    
    vol_val = _get_val("Volume") 
    volume = vol_val if vol_val is not None else np.ones(len(close))
    
    open_p = _get_val("Open")
    open_val = open_p if open_p is not None else close
    n = len(close)
    eps = 1e-8
    feats = []

    def safe_pct(a, b):
        return (a - b) / (np.abs(b) + eps)

    # 1. Log return
    log_ret = np.zeros(n)
    log_ret[1:] = np.diff(np.log(close + eps))
    feats.append(log_ret)

    # 2-4. Short-term momentum
    for lag in [1, 3, 5]:
        mom = np.zeros(n)
        mom[lag:] = safe_pct(close[lag:], close[:-lag])
        feats.append(mom)

    # 5. HL range normalised
    hl = (high - low) / (close + eps)
    feats.append(hl)

    # 6. Close vs open proximity
    feats.append(safe_pct(close, open_val))

    # 7-9. SMA ratios
    for w in [5, 10, 20]:
        sma = pd.Series(close).rolling(w, min_periods=1).mean().values
        feats.append(safe_pct(close, sma))

    # 10. Volatility (rolling std)
    vol5 = pd.Series(log_ret).rolling(5, min_periods=1).std().values
    feats.append(vol5)

    # 11. RSI proxy
    delta = np.concatenate([[0.0], np.diff(close)])
    up    = np.where(delta > 0, delta, 0)
    down  = np.where(delta < 0, -delta, 0)
    avg_u = pd.Series(up).rolling(14, min_periods=1).mean().values
    avg_d = pd.Series(down).rolling(14, min_periods=1).mean().values
    rs    = avg_u / (avg_d + eps)
    rsi   = 1.0 - 1.0 / (1.0 + rs)
    feats.append(rsi)

    # 12. Volume change
    vol_chg = np.zeros(n)
    vol_chg[1:] = safe_pct(volume[1:], volume[:-1])
    feats.append(vol_chg)

    # 13. High/Close
    feats.append(safe_pct(high, close))

    # 14. Low/Close
    feats.append(safe_pct(low, close))

    # 15. 10-day return
    ret10 = np.zeros(n)
    ret10[10:] = safe_pct(close[10:], close[:-10])
    feats.append(ret10)

    matrix = np.stack(feats, axis=1)   # (n, N_FEATURES)
    # Replace NaN/Inf with 0 then clip extremes
    matrix = np.nan_to_num(matrix.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -5, 5)
    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# MiniMASTER Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    # ---------------------------------------------------------------------------
    # Gate Module — core MASTER paper contribution (AAAI-2024)
    # Applies a softmax-scaled per-feature weight using the last bar's features
    # as the "market information" guide (Section 3.2 of the paper).
    # ---------------------------------------------------------------------------
    class Gate(nn.Module):
        """
        Soft feature-selection gate from the MASTER paper.
        Uses the last time-step's raw features to produce a per-channel
        attention weight via a temperature-scaled softmax.
        d_input  -> number of raw features
        d_output -> number of raw features (same, we weight them)
        beta     -> temperature (higher = softer / more uniform weights)
        """
        def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
            super().__init__()
            self.trans    = nn.Linear(d_input, d_output)
            self.d_output = d_output
            self.beta     = beta

        def forward(self, gate_input):           # gate_input: (B, d_input)
            out = self.trans(gate_input)          # (B, d_output)
            out = torch.softmax(out / self.beta, dim=-1)
            return self.d_output * out            # (B, d_output)

    class MiniMASTER(nn.Module):
        """
        MASTER-style transformer for single-stock prediction.
        Input:  (batch, seq_len, n_features)
        Output: (batch,) — predicted normalised next-day return

        Key difference vs plain transformer:
          • A Gate module (from the MASTER paper) weighs each input feature
            using the LAST bar's feature vector before the transformer encoder.
            This recovers the paper's "market-guided feature selection" idea.
        """
        def __init__(self, n_features=N_FEATURES, hidden=HIDDEN_DIM,
                     n_heads=N_HEADS, n_layers=N_LAYERS, gate_beta=1.0):
            super().__init__()
            # Gate: takes last-bar features -> per-feature weight
            self.feature_gate = Gate(n_features, n_features, beta=gate_beta)

            # Project gated features to hidden dim
            self.input_proj = nn.Linear(n_features, hidden)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                dropout=0.1, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):                    # x: (B, T, F)
            # 1. Per-sequence z-score normalisation (prevents gradient explosion)
            mean = x.mean(dim=1, keepdim=True)   # (B, 1, F)
            std  = x.std(dim=1, keepdim=True) + 1e-6
            x    = (x - mean) / std              # (B, T, F)

            # 2. Gate — use last bar's normed features to weight every bar
            gate_input  = x[:, -1, :]            # (B, F)
            gate_weight = self.feature_gate(gate_input).unsqueeze(1)  # (B, 1, F)
            x = x * gate_weight                  # (B, T, F)  element-wise

            # 3. Project → Transformer encoder → last token → head
            x = self.input_proj(x)               # (B, T, hidden)
            x = self.transformer(x)              # (B, T, hidden)
            x = x[:, -1, :]                      # (B, hidden)
            return self.head(x).squeeze(-1)      # (B,)
else:
    MiniMASTER = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_sequences(feats: np.ndarray, close: np.ndarray, seq_len=SEQ_LEN):
    """
    Build (X, y) where y = next-day log return.
    Uses vectorised numpy stride tricks instead of a Python loop to avoid
    a RAM spike when building thousands of overlapping windows.
    """
    n = len(feats)
    # Aligned log_ret: log_ret[i] is the return from day i-1 to day i.
    log_ret = np.zeros(n, dtype=np.float32)
    log_ret[1:] = np.diff(np.log(close.astype(np.float64) + 1e-8)).astype(np.float32)

    if n <= seq_len:
        return np.empty((0, seq_len, feats.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Stride trick: create a (n-seq_len, seq_len, n_features) view with zero copy
    shape   = (n - seq_len, seq_len, feats.shape[1])
    strides = (feats.strides[0], feats.strides[0], feats.strides[1])
    X = np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()
    y = log_ret[seq_len:]
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# Train / Load
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray):
    """Quick training run on available hardware (CPU-safe)."""
    if not TORCH_AVAILABLE:
        return None

    import os, torch
    from torch.utils.data import TensorDataset, DataLoader

    model = MiniMASTER().to(DEVICE)

    # Try loading from previous checkpoint
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[MASTER] Loaded existing weights for incremental training.")
        except Exception as e:
            print(f"[MASTER] Could not load weights ({e}), training from scratch.")

    Xt = torch.tensor(X).to(DEVICE)
    yt = torch.tensor(y).to(DEVICE)

    # Remove any NaN/Inf rows from training data
    valid_mask = torch.isfinite(Xt).all(dim=-1).all(dim=-1) & torch.isfinite(yt)
    Xt = Xt[valid_mask]
    yt = yt[valid_mask]
    if len(Xt) < 10:
        print("[MASTER] Too many NaN samples, skipping training.")
        return None

    # Scale log returns to percentages for numerical stability during training
    # (Avoids small gradients/loss with Adam)
    yt_pct = yt * 100.0

    ds = TensorDataset(Xt, yt_pct)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    loss_fn   = nn.MSELoss()
    
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    final_loss = 0.0

    model.train()
    for epoch in range(MAX_EPOCHS):
        total_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                continue   # skip bad batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / max(len(dl), 1)
        final_loss = epoch_loss
        scheduler.step()
        
        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print(f"[MASTER] Early stopping triggered at epoch {epoch+1}")
                early_stop = True
                break
                
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[MASTER] Epoch {epoch+1}/{MAX_EPOCHS}  loss={epoch_loss:.6f}")

    print(f"[MASTER] Final Train Loss: {final_loss:.6f} | Early Stopped: {early_stop}")

    # Save checkpoint for next run
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[MASTER] Weights saved to {MODEL_PATH}")
    except Exception as e:
        print(f"[MASTER] Could not save weights: {e}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference / pipeline
# ---------------------------------------------------------------------------

def run_master_pipeline(df: pd.DataFrame) -> dict | None:
    """
    Full MASTER pipeline:
      1. Engineer features
      2. Train / load model
      3. Predict next 5 days
      4. Return signal dict
    """
    try:
        print("[MASTER] Engineering features...")
        feats = engineer_features(df)
        close = df["Close"].values.astype(float)

        if TORCH_AVAILABLE:
            X, y = build_sequences(feats, close)
            if len(X) < 50:
                print("[MASTER] Not enough data for training.")
                return None

            print(f"[MASTER] Training on {len(X)} sequences (max {MAX_EPOCHS} epochs)...")
            model = train_model(X, y)
            if model is None:
                return None

            model.eval()
            with torch.no_grad(): # Disable VRAM heavy gradient tracking
                # Use last SEQ_LEN bars as input
                last_seq = torch.tensor(feats[-SEQ_LEN:]).unsqueeze(0).to(DEVICE)
                pred_ret_tensor = model(last_seq)
                pred_ret_pct = float(pred_ret_tensor.item()) if hasattr(pred_ret_tensor, "item") else 0.0
                pred_ret_pct = max(pred_ret_pct, -99.0) # Prevent log(<=0)
                pred_ret_log = np.log(1 + pred_ret_pct / 100.0)  # Convert back to log return for price forecasting
                
                # VERY IMPORTANT: explicitly delete GPU variables and dump VRAM immediately
                del last_seq, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            # Fallback: momentum-based rule
            recent_ret = np.diff(np.log(close[-10:] + 1e-8)).mean()
            pred_ret_log = recent_ret
            pred_ret_pct = (np.exp(pred_ret_log) - 1) * 100

        # Determine signal based on improved thresholds
        if pred_ret_pct > 0.3:
            signal = "BUY"
        elif pred_ret_pct < -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"
        confidence = min(abs(pred_ret_pct) / 2.0, 1.0)

        # Simple linear 5-day forecast
        last_price = close[-1]
        daily_step = np.exp(pred_ret_log)
        forecast   = [last_price * (daily_step ** d) for d in range(1, PRED_DAYS + 1)]

        return {
            "signal":               signal,
            "confidence":           confidence,
            "predicted_return_pct": round(pred_ret_pct, 3),
            "forecast_prices":      [round(p, 2) for p in forecast],
        }

    except Exception as e:
        print(f"[MASTER] Pipeline error: {e}")
        return None

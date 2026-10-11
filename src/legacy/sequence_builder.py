"""
Turn the per-window feature table into forecasting sequences.

This is the key reframing that makes it "forecasting" instead of "detection":

    Input:  windows[i-lookback : i]              (past W windows of features)
    Label:  1 if any window in [i, i+horizon) is an attack window, else 0

So the model never sees the window it's predicting about — only the lead-up to
it. A true positive means we raised the alarm *before* the attack window's own
features would have given it away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules (features.py ...) live in src/
from features import FEATURE_COLUMNS


def build_sequences(
    windows: pd.DataFrame,
    lookback: int = 12,
    horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X       : (n_samples, lookback, n_features) float array
        y        : (n_samples,) binary forecast label
        start_idx: (n_samples,) the window_id each sample's lookback window ends at,
                   useful for plotting predictions back against time later
    """
    feats = windows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    is_attack = windows["window_is_attack"].to_numpy()

    # Normalize features (simple z-score; fit stats are returned by caller if needed)
    mu, sigma = feats.mean(axis=0), feats.std(axis=0) + 1e-6
    feats_norm = (feats - mu) / sigma

    n = len(windows)
    X, y, start_idx = [], [], []
    for i in range(lookback, n - horizon):
        window_seq = feats_norm[i - lookback : i]
        future = is_attack[i : i + horizon]
        label = int(future.any())
        X.append(window_seq)
        y.append(label)
        start_idx.append(i)

    return np.array(X), np.array(y), np.array(start_idx)


def train_val_split_by_time(X, y, start_idx, val_fraction: float = 0.2):
    """Time-ordered split (never shuffle — forecasting must respect causality)."""
    n = len(X)
    split = int(n * (1 - val_fraction))
    return (
        X[:split], y[:split], start_idx[:split],
        X[split:], y[split:], start_idx[split:],
    )


if __name__ == "__main__":
    windows = pd.read_parquet("data/windows.parquet")
    X, y, idx = build_sequences(windows)
    print("X:", X.shape, "y positive rate:", y.mean())

"""
Training loop for the attack-forecasting LSTM.

Reports the metric that actually matters for this problem — not just
PR-AUC, but *lead time*: for attacks the model caught, how many seconds
before the attack window did the alarm cross threshold?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import AttackForecastLSTM
from sequence_builder import build_sequences, train_val_split_by_time


def train(
    windows_path: str = "data/windows.parquet",
    lookback: int = 12,
    horizon: int = 6,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
):
    windows = pd.read_parquet(windows_path)
    X, y, start_idx = build_sequences(windows, lookback=lookback, horizon=horizon)
    X_tr, y_tr, idx_tr, X_val, y_val, idx_val = train_val_split_by_time(X, y, start_idx)

    print(f"Train: {len(X_tr)} samples, positive rate {y_tr.mean():.3f}")
    print(f"Val:   {len(X_val)} samples, positive rate {y_val.mean():.3f}")

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = AttackForecastLSTM(n_features=X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Class imbalance is likely severe (attacks are rare) — weight the positive class.
    pos_weight = torch.tensor([(1 - y_tr.mean()) / max(y_tr.mean(), 1e-3)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val).to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        ap = average_precision_score(y_val, val_probs) if y_val.sum() > 0 else float("nan")
        auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else float("nan")
        print(f"Epoch {epoch+1:2d}  loss={total_loss/len(train_ds):.4f}  val_PR-AUC={ap:.3f}  val_ROC-AUC={auc:.3f}")

    torch.save(model.state_dict(), "models/attack_forecast_lstm.pt")
    np.savez(
        "models/val_predictions.npz",
        probs=val_probs, y=y_val, window_idx=idx_val,
    )
    print("Saved model to models/attack_forecast_lstm.pt")
    report_lead_time(windows, val_probs, y_val, idx_val, horizon)


def report_lead_time(windows, probs, y_val, idx_val, horizon, threshold=0.5):
    """
    For each true-positive window sequence, find how many windows before the
    actual attack onset the alarm first crossed threshold within that sample's
    forecast horizon. This is the number that tells you if 'forecasting' is
    actually working, not just 'detecting with a delay'.
    """
    attack_rows = windows["window_is_attack"].to_numpy()
    lead_times = []
    for i, (p, label, idx) in enumerate(zip(probs, y_val, idx_val)):
        if label == 1 and p >= threshold:
            future = attack_rows[idx : idx + horizon]
            onset_offset = int(np.argmax(future))  # first attack window in horizon
            lead_times.append(onset_offset)

    if lead_times:
        print(
            f"\nLead time (windows before attack onset, among caught attacks): "
            f"mean={np.mean(lead_times):.1f}  median={np.median(lead_times):.1f}  n={len(lead_times)}"
        )
    else:
        print("\nNo true positives at this threshold to compute lead time from.")


if __name__ == "__main__":
    train()

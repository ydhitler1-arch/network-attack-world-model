"""
LSTM sequence classifier: takes a window of recent traffic features and
outputs a single risk score (probability an attack starts within the forecast
horizon).

Deliberately small — this is meant to be understandable end-to-end, not a
leaderboard-chasing architecture. Swap in a Transformer encoder later once
this baseline is working and you understand its failure modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttackForecastLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — final layer's final hidden state
        logit = self.head(last_hidden).squeeze(-1)  # (batch,)
        return logit  # raw logit; apply sigmoid for probability

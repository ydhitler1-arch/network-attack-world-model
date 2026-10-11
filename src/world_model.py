"""
World model of network state: learns P(S_{t+1} | S_<=t) and reads infiltration risk and
ATT&CK stage off the (real or imagined) state.

One recurrent cell (LSTM) consumes a state x_tau and produces, from its hidden state h_tau:
    dynamics   Gaussian over the next state:  x_{tau+1} ~ N(x_tau + delta(h), diag(sigma(h)^2))
    infiltration logit  P(window tau is attack | history up to tau)
    stage logits        ATT&CK stage of window tau (class 0 = benign)

K-step rollout: after the L real windows, the predicted mean of the next state is fed back as the
next input, K times. The infiltration/stage heads are read at every step, so the model outputs a
probability time series for the current window (k=0) and the next K windows (k=1..K), each
computed from an *imagined* state. Training unrolls the same rollout (src/wm_train.py), so the
dynamics are trained on multi-step consistency, not only one-step prediction.

Ablation: use_dynamics=False drops the dynamics head (and training loss); the model is then a
plain LSTM classifier on the history (no rollout), which Phase 3 uses as the "LSTM classifier"
comparison.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0


def _norm_label(s: str) -> str:
    """Make labels ASCII-comparable (the CIC files use an en dash in 'Web Attack – ...')."""
    return re.sub(r"[^\x00-\x7f]", "-", str(s)).strip()


def stage_ids(attack_type: pd.Series, cfg: dict) -> np.ndarray:
    """Window attack_type ('' for benign) -> stage index per cfg['stages']."""
    stages = cfg["stages"]
    mp = {_norm_label(k): stages.index(v) for k, v in cfg["label_to_stage"].items()}
    out = np.zeros(len(attack_type), dtype=np.int64)
    for i, a in enumerate(attack_type.fillna("").to_numpy()):
        a = _norm_label(a)
        if a and a != "BENIGN":
            if a not in mp:
                raise KeyError(f"attack label {a!r} has no stage in cfg['label_to_stage']")
            out[i] = mp[a]
    return out


class WorldModel(nn.Module):
    def __init__(self, n_features: int, n_stages: int, hidden: int = 128, dropout: float = 0.1,
                 use_dynamics: bool = True):
        super().__init__()
        self.n_features, self.n_stages, self.use_dynamics = n_features, n_stages, use_dynamics
        self.cell = nn.LSTMCell(n_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.dyn_delta = nn.Linear(hidden, n_features)
        self.dyn_logvar = nn.Linear(hidden, n_features)
        self.atk = nn.Linear(hidden, 1)
        self.stage = nn.Linear(hidden, n_stages)
        nn.init.zeros_(self.dyn_delta.weight)  # start at "next state = current state"
        nn.init.zeros_(self.dyn_delta.bias)

    def step(self, x, state):
        h, c = self.cell(x, state)
        hd = self.drop(h)
        return {
            "mu": x + self.dyn_delta(hd),
            "logvar": self.dyn_logvar(hd).clamp(LOGVAR_MIN, LOGVAR_MAX),
            "atk": self.atk(hd).squeeze(-1),
            "stage": self.stage(hd),
            "h": h,
        }, (h, c)

    def forward(self, x_hist: torch.Tensor, K: int = 0):
        """x_hist (B, L, D) real states. Returns tensors with T = L + K steps on dim 1.
        Step j<L consumes real x_j; step j>=L consumes the imagined mean from step j-1."""
        outs, state = [], None
        for j in range(x_hist.shape[1]):
            o, state = self.step(x_hist[:, j], state)
            outs.append(o)
        if K:
            if not self.use_dynamics:
                raise ValueError("rollout needs use_dynamics=True")
            for _ in range(K):
                o, state = self.step(outs[-1]["mu"], state)
                outs.append(o)
        return {k: torch.stack([o[k] for o in outs], dim=1) for k in outs[0]}


# ---------------------------------------------------------------- save / load / predict

def save_model(path, model: WorldModel, cfg: dict, cols: list[str], scaler_json: dict, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "cols": cols, "scaler": scaler_json,
                "extra": extra or {}}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = WorldModel(len(ck["cols"]), len(cfg["stages"]), cfg["hidden"], cfg["dropout"], cfg.get("use_dynamics", True))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def scaler_from_json(d: dict):
    from state import StateScaler
    s = StateScaler(d["cols"])
    s.mean_, s.std_ = np.array(d["mean"]), np.array(d["std"])
    return s


def history_index(ck: dict, state: pd.DataFrame):
    """Normalised states Z (T,D), index matrix (T,L) of each window's history (padded by repeating the
    first window of its segment), and valid (T,) = full L-window history available."""
    cfg = ck["cfg"]
    L = cfg["lookback"]
    scaler = scaler_from_json(ck["scaler"])
    Z = np.clip(scaler.transform(state), -cfg["clip_z"], cfg["clip_z"]).astype(np.float32)
    seg = state["segment"].to_numpy() if "segment" in state else np.zeros(len(state), dtype=int)
    first = pd.Series(np.arange(len(Z))).groupby(seg).transform("min").to_numpy()
    idx = np.arange(len(Z))[:, None] + np.arange(-L + 1, 1)[None, :]
    valid = idx[:, 0] >= first
    return Z, np.maximum(idx, first[:, None]), valid


@torch.no_grad()
def predict_timeline(model: WorldModel, ck: dict, state: pd.DataFrame, K: int | None = None, batch: int = 512):
    """Run the model on a window table (columns incl. ck['cols'], optional 'segment').

    For every window t with L-1 earlier windows in the same segment (earlier ones are padded by
    repeating the first window, flagged valid=False) returns:
        p_infil   (T, K+1)   P(attack) for windows t, t+1, ..., t+K   (index 0 = current window)
        stage_p   (T, K+1, S)
        mu_future (T, K, D)  predicted normalised states for t+1..t+K
        valid     (T,)
    """
    cfg = ck["cfg"]
    L, K = cfg["lookback"], (cfg["horizon"] if K is None else K)
    Z, idx, valid = history_index(ck, state)
    T, D = Z.shape
    model.eval()
    P, S, M = [], [], []
    for a in range(0, T, batch):
        x = torch.from_numpy(Z[idx[a:a + batch]])
        o = model(x, K)
        P.append(torch.sigmoid(o["atk"][:, L - 1:]).numpy())
        S.append(torch.softmax(o["stage"][:, L - 1:], dim=-1).numpy())
        M.append(o["mu"][:, L - 1:L - 1 + K].numpy() if K else np.zeros((len(x), 0, D), np.float32))
    return {"p_infil": np.concatenate(P), "stage_p": np.concatenate(S),
            "mu_future": np.concatenate(M), "valid": valid, "Z": Z}

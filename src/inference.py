"""
Inference pipeline behind the offline demo (no Streamlit imports, so it can be tested and scripted).

    flows (CSV or PCAP) -> 60 s state windows -> world-model rollout -> infiltration probabilities,
    ATT&CK stage, SHAP explanations, and counterfactual "which source's flows drive this alert".

Everything runs locally from files in models/ (no network access).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import load_csv_file
from explain import explain_logreg, explain_world_model, fit_logreg
from state import STATE_COLUMNS, build_state
from world_model import history_index, load_model, predict_timeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "world_model_full.pt"
BG_PATH = ROOT / "models" / "explain_background.npz"
STATE_TRAIN = ROOT / "data" / "state_v1.parquet"


# ------------------------------------------------------------------ loading

def load_flows(path: str | Path, kind: str | None = None, max_packets: int | None = 3_000_000) -> pd.DataFrame:
    """CSV (CICFlowMeter layout, or the demo's canonical layout; .csv / .csv.gz) or PCAP -> flows."""
    path = Path(path)
    kind = kind or ("pcap" if path.suffix.lower() in (".pcap", ".pcapng", ".cap") else "csv")
    if kind == "pcap":
        from pcap_to_flows import pcap_to_flows
        return pcap_to_flows(str(path), max_packets=max_packets)
    return load_csv_file(path, extended=True)


# ------------------------------------------------------------------ analysis

class Analysis:
    """Holds one analysed capture. Heavy objects (model, background) are loaded once per process."""

    def __init__(self, flows, state, model, ck, pred, ws, bg):
        self.flows, self.state, self.model, self.ck, self.pred, self.ws, self.bg = flows, state, model, ck, pred, ws, bg
        self.K = ck["cfg"]["horizon"]
        self.stages = ck["cfg"]["stages"]
        self.t = pd.to_datetime(state["window_start"], unit="s")

    @property
    def p_now(self):
        return self.pred["p_infil"][:, 0]


_MODEL_CACHE: dict = {}


def get_model(path=MODEL_PATH):
    if str(path) not in _MODEL_CACHE:
        _MODEL_CACHE[str(path)] = load_model(path)
    return _MODEL_CACHE[str(path)]


def analyze(flows: pd.DataFrame, window_seconds: float = 60.0, model_path=MODEL_PATH) -> Analysis:
    flows = flows.sort_values("timestamp").reset_index(drop=True)
    state = build_state(flows, window_seconds=window_seconds)
    model, ck = get_model(model_path)
    pred = predict_timeline(model, ck, state)
    bg = np.load(BG_PATH)["bg"]
    return Analysis(flows, state, model, ck, pred, window_seconds, bg)


def default_thresholds() -> dict:
    """Alert thresholds at 2% of benign training windows (in-sample, so optimistic: Phase 3 measured 6%
    realised false alarms for the world model on unseen days). Also fits the logistic-regression baseline."""
    st = pd.read_parquet(STATE_TRAIN).reset_index(drop=True)
    model, ck = get_model()
    pred = predict_timeline(model, ck, st)
    ok = pred["valid"] & (st["window_is_attack"].to_numpy() == 0)
    wm_thr = float(np.quantile(pred["p_infil"][ok, 0], 0.98))
    Z = pred["Z"]
    y = st["window_is_attack"].to_numpy()
    clf = fit_logreg(Z, y)
    lr_thr = float(np.quantile(clf.decision_function(Z)[ok], 0.98))
    return {"world_model": wm_thr, "logreg": lr_thr, "clf": clf, "bg_mean": Z[ok].mean(0)}


def logreg_scores(an: Analysis, clf) -> np.ndarray:
    return clf.decision_function(an.pred["Z"])


# ------------------------------------------------------------------ explanations

def explain_windows(an: Analysis, rows, n_samples: int = 128) -> dict:
    """SHAP for the given window rows (any number); arrays are aligned with `rows`."""
    return explain_world_model(an.model, an.ck, an.state, an.bg, n_samples=n_samples, rows=rows)


def top_drivers(phi_feat_row: np.ndarray, z_row: np.ndarray, raw_row: np.ndarray, top: int = 5) -> pd.DataFrame:
    order = np.argsort(-np.abs(phi_feat_row))[:top]
    return pd.DataFrame({
        "feature": [STATE_COLUMNS[j] for j in order], "shap": phi_feat_row[order],
        "value": raw_row[order], "z": z_row[order]})


def flows_in_window(an: Analysis, t: int) -> pd.DataFrame:
    a, b = an.state["window_start"].iloc[t], an.state["window_end"].iloc[t]
    ts = an.flows["timestamp"].to_numpy()
    lo, hi = np.searchsorted(ts, [a, b])  # flows are sorted by timestamp
    return an.flows.iloc[lo:hi]


@torch.no_grad()
def p_for_state(an: Analysis, state: pd.DataFrame, t: int) -> np.ndarray:
    """Infiltration probabilities (k=0..K) for window t of a (possibly modified) state table."""
    Z, idx, _ = history_index(an.ck, state)
    L = an.ck["cfg"]["lookback"]
    x = torch.from_numpy(Z[idx[t:t + 1]])
    o = an.model(x, an.K)
    return torch.sigmoid(o["atk"][0, L - 1:]).numpy()


def suspect_hosts(an: Analysis, t: int, top: int = 6) -> pd.DataFrame:
    """Counterfactual attribution to flows: remove every flow that involves one host (as source or
    destination; a connection is often recorded in both directions), recompute window t's state and
    re-score. A large drop means that host's traffic drives the alert."""
    fl = flows_in_window(an, t)
    if fl.empty:
        return pd.DataFrame()
    base_p = float(an.pred["p_infil"][t, 0])
    as_src, as_dst = fl["src_ip"].value_counts(), fl["dst_ip"].value_counts()
    involved = as_src.add(as_dst, fill_value=0).sort_values(ascending=False)
    by_ports = fl.groupby("src_ip")["dst_port"].nunique().sort_values(ascending=False)
    cands = list(dict.fromkeys(list(involved.index[: top // 2 + 1]) + list(by_ports.index[: top // 2 + 1])))[:top]
    rows = []
    L = an.ck["cfg"]["lookback"]
    for ip in cands:
        mask = (fl["src_ip"] == ip) | (fl["dst_ip"] == ip)
        # remove the host from the whole lookback: the model carries state from earlier windows, so
        # removing it from window t alone would leave the history that still says "attack"
        st = an.state.copy()
        for h in range(max(0, t - L + 1), t + 1):
            fh = flows_in_window(an, h) if h != t else fl
            rest = fh[(fh["src_ip"] != ip) & (fh["dst_ip"] != ip)]
            if rest.empty:
                st.loc[st.index[h], STATE_COLUMNS] = 0.0
            else:
                new = build_state(rest, window_seconds=an.ws).iloc[0]
                st.loc[st.index[h], STATE_COLUMNS] = new[STATE_COLUMNS].to_numpy(float)
        p_wo = float(p_for_state(an, st, t)[0])
        mine = fl[mask]
        sent = mine[mine["src_ip"] == ip]
        rows.append({"host": ip, "flows": len(mine), "as_source": len(sent), "as_destination": len(mine) - len(sent),
                     "share_of_window": len(mine) / len(fl),
                     "distinct_dst_ports_as_source": int(sent["dst_port"].nunique()),
                     "P_without_host": p_wo, "drop": base_p - p_wo,
                     **({"labels": ",".join(sorted(set(mine["label"]) - {"UNLABELLED"}))}
                        if (mine["label"] != "UNLABELLED").any() else {})})
    return pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)


def flow_table(an: Analysis, t: int, host: str, limit: int = 500) -> pd.DataFrame:
    fl = flows_in_window(an, t)
    fl = fl[(fl["src_ip"] == host) | (fl["dst_ip"] == host)].head(limit)
    out = pd.DataFrame({
        "time": pd.to_datetime(fl["timestamp"], unit="s").dt.strftime("%H:%M:%S"),
        "src": fl["src_ip"] + ":" + fl["src_port"].astype(int).astype(str),
        "dst": fl["dst_ip"] + ":" + fl["dst_port"].astype(int).astype(str),
        "proto": fl["protocol"], "fwd_pkts": fl["fwd_packets"], "bwd_pkts": fl["bwd_packets"],
        "bytes": fl["fwd_bytes"] + fl["bwd_bytes"], "duration_ms": (fl["duration"] / 1e3).round(1),
        "SYN": fl["syn_flag"], "ACK": fl["ack_flag"], "RST": fl["rst_flag"], "label": fl["label"]})
    return out.reset_index(drop=True)


def stage_summary(an: Analysis, t: int) -> pd.DataFrame:
    p = an.pred["stage_p"][t, 0]
    return pd.DataFrame({"stage": an.stages, "probability": p}).sort_values("probability", ascending=False)

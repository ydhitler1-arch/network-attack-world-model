"""
Network state vector S_t for the world model (Phase 1).

One row = one time window of flows. Features come in three groups:

  base      the 13 original window features (src/features.py, unchanged)
  flow      TCP flag shares, protocol mix, packets/bytes per flow, IAT stats, bidirectional
            ratios, scan signatures (no-reply share, single-packet share, ports per source)
  packet*   packet-level PROXIES derivable from CICFlowMeter CSV columns: TCP window of the
            first packet, per-flow average-packet-size distribution.

* A CSV dataset cannot provide TTL (+variance), IP fragment flags, retransmissions, or a true
  payload-size distribution; those need packets. src/pcap_to_flows.py extracts them as extra
  columns (PACKET_ONLY_COLUMNS). They are NOT part of the core state because the same model
  must run on CSV input. See docs/FEATURES.md.

Usage:
    python src/state.py data/raw 60
writes data/state_v1.parquet (raw aggregates + metadata), data/state_v1_scaled.parquet
(normalised matrix) and models/state_scaler.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS as BASE_COLUMNS
from features import build_windows

FLOW_COLUMNS = [
    "mean_pkts_per_flow", "mean_bytes_per_flow",
    "fin_ratio", "rst_ratio", "psh_ratio", "ack_ratio", "urg_ratio", "syn_only_ratio",
    "tcp_share", "udp_share",
    "log_iat_mean", "log_iat_std", "log_iat_max",
    "pkt_ratio_bwd_fwd", "byte_ratio_bwd_fwd", "mean_down_up_ratio",
    "no_reply_ratio", "single_pkt_ratio",
    "max_ports_per_src", "max_dsts_per_src", "dst_port_entropy", "src_port_uniq_ratio",
]
PACKET_PROXY_COLUMNS = [
    "mean_log_init_win_fwd", "std_log_init_win_fwd", "win_missing_ratio",
    "mean_avg_pkt_size", "std_avg_pkt_size", "small_pkt_ratio", "large_pkt_ratio",
    "mean_pkt_len_std",
]
STATE_COLUMNS = BASE_COLUMNS + FLOW_COLUMNS + PACKET_PROXY_COLUMNS

# Only available when the input is a PCAP (per-flow columns aggregated per window).
PACKET_ONLY_COLUMNS = [
    "mean_ttl", "ttl_var_mean", "tcp_win_mean", "frag_ratio", "retrans_ratio", "payload_std_mean",
]

# Heavy-tailed count/rate features get log1p before z-scoring.
LOG_COLUMNS = [
    "flow_count", "packet_rate", "byte_rate", "unique_dst_ips", "unique_dst_ports", "unique_src_ips",
    "mean_pkts_per_flow", "mean_bytes_per_flow", "max_ports_per_src", "max_dsts_per_src",
    "mean_avg_pkt_size", "std_avg_pkt_size", "pkt_ratio_bwd_fwd", "byte_ratio_bwd_fwd",
    "mean_down_up_ratio",
]

SMALL_PKT_BYTES = 100    # average packet size below this = "small" (scan probes, ACK-only, floods)
LARGE_PKT_BYTES = 1000   # above this = bulk transfer


def _entropy_per_window(w: pd.Series, v: pd.Series) -> pd.Series:
    counts = pd.DataFrame({"w": w.to_numpy(), "v": v.to_numpy()}).groupby(["w", "v"]).size()
    p = counts / counts.groupby(level=0).transform("sum")
    return -(p * np.log2(p)).groupby(level=0).sum()


def _extra_features(df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Per-window flow-level and packet-proxy features. Index = window_id."""
    t0 = df["timestamp"].min()  # must match build_windows
    w = ((df["timestamp"] - t0) // window_seconds).astype(int)
    f32 = np.float32
    fwd_p, bwd_p = df["fwd_packets"].to_numpy(f32), df["bwd_packets"].to_numpy(f32)
    pk = fwd_p + bwd_p
    by = (df["fwd_bytes"] + df["bwd_bytes"]).to_numpy(f32)
    proto = pd.to_numeric(df["protocol"], errors="coerce").fillna(-1)
    iw = df["init_win_fwd"].where(df["init_win_fwd"] >= 0)

    t = pd.DataFrame({
        "w": w.to_numpy(),
        "pk": pk, "by": by, "fwd_p": fwd_p, "bwd_p": bwd_p,
        "fwd_b": df["fwd_bytes"].to_numpy(f32), "bwd_b": df["bwd_bytes"].to_numpy(f32),
        "fin": (df["fin_flag"] > 0).to_numpy(f32), "rst": (df["rst_flag"] > 0).to_numpy(f32),
        "psh": (df["psh_flag"] > 0).to_numpy(f32), "ack": (df["ack_flag"] > 0).to_numpy(f32),
        "urg": (df["urg_flag"] > 0).to_numpy(f32),
        "syn_only": ((df["syn_flag"] > 0) & (df["ack_flag"] == 0)).to_numpy(f32),
        "tcp": (proto == 6).to_numpy(f32), "udp": (proto == 17).to_numpy(f32),
        "iat_mean": np.log1p(df["flow_iat_mean"].clip(lower=0)).to_numpy(f32),
        "iat_std": np.log1p(df["flow_iat_std"].clip(lower=0)).to_numpy(f32),
        "iat_max": np.log1p(df["flow_iat_max"].clip(lower=0)).to_numpy(f32),
        "down_up": df["down_up_ratio"].to_numpy(f32),
        "no_reply": (df["bwd_packets"] == 0).to_numpy(f32),
        "single": (pk <= 2).astype(f32),
        "log_iw": np.log1p(iw).to_numpy(f32),
        "win_missing": (df["init_win_fwd"] < 0).to_numpy(f32),
        "aps": df["avg_pkt_size"].clip(lower=0).to_numpy(f32),
        "small": (df["avg_pkt_size"] < SMALL_PKT_BYTES).to_numpy(f32),
        "large": (df["avg_pkt_size"] >= LARGE_PKT_BYTES).to_numpy(f32),
        "pls": df["pkt_len_std"].clip(lower=0).to_numpy(f32),
    })
    g = t.groupby("w")
    s = g.sum()
    n = g.size()
    out = pd.DataFrame(index=s.index)
    out["mean_pkts_per_flow"] = s["pk"] / n
    out["mean_bytes_per_flow"] = s["by"] / n
    for k in ("fin", "rst", "psh", "ack", "urg"):
        out[f"{k}_ratio"] = s[k] / n
    out["syn_only_ratio"] = s["syn_only"] / n
    out["tcp_share"] = s["tcp"] / n
    out["udp_share"] = s["udp"] / n
    out["log_iat_mean"] = g["iat_mean"].mean()
    out["log_iat_std"] = g["iat_std"].mean()
    out["log_iat_max"] = g["iat_max"].mean()
    out["pkt_ratio_bwd_fwd"] = s["bwd_p"] / s["fwd_p"].clip(lower=1)
    out["byte_ratio_bwd_fwd"] = s["bwd_b"] / s["fwd_b"].clip(lower=1)
    out["mean_down_up_ratio"] = g["down_up"].mean()
    out["no_reply_ratio"] = s["no_reply"] / n
    out["single_pkt_ratio"] = s["single"] / n
    # Scan signatures: one source touching many ports (vertical) or hosts (horizontal).
    key = pd.DataFrame({"w": w.to_numpy(), "src": df["src_ip"].to_numpy()})
    key["dport"] = df["dst_port"].to_numpy()
    key["dip"] = df["dst_ip"].to_numpy()
    out["max_ports_per_src"] = key.groupby(["w", "src"])["dport"].nunique().groupby(level=0).max()
    out["max_dsts_per_src"] = key.groupby(["w", "src"])["dip"].nunique().groupby(level=0).max()
    out["dst_port_entropy"] = _entropy_per_window(w, df["dst_port"])
    out["src_port_uniq_ratio"] = (
        pd.DataFrame({"w": w.to_numpy(), "p": df["src_port"].to_numpy()}).groupby("w")["p"].nunique() / n
    )
    # Packet-level proxies (CSV): first-packet TCP window, avg-packet-size distribution.
    out["mean_log_init_win_fwd"] = g["log_iw"].mean()
    out["std_log_init_win_fwd"] = g["log_iw"].std()
    out["win_missing_ratio"] = s["win_missing"] / n
    out["mean_avg_pkt_size"] = g["aps"].mean()
    out["std_avg_pkt_size"] = g["aps"].std()
    out["small_pkt_ratio"] = s["small"] / n
    out["large_pkt_ratio"] = s["large"] / n
    out["mean_pkt_len_std"] = g["pls"].mean()

    if "ttl_mean" in df.columns:  # PCAP-derived flows: true packet-level features
        p = pd.DataFrame({
            "w": w.to_numpy(),
            "ttl": df["ttl_mean"].to_numpy(f32), "ttlv": df["ttl_var"].to_numpy(f32),
            "win": df["tcp_win_mean"].to_numpy(f32),
            "frag": (df["frag_count"] > 0).to_numpy(f32),
            "retx": (df["retrans_count"] > 0).to_numpy(f32), "pls": df["payload_std"].to_numpy(f32),
        }).groupby("w")
        out["mean_ttl"] = p["ttl"].mean()
        out["ttl_var_mean"] = p["ttlv"].mean()
        out["tcp_win_mean"] = p["win"].mean()
        out["frag_ratio"] = p["frag"].mean()
        out["retrans_ratio"] = p["retx"].mean()
        out["payload_std_mean"] = p["pls"].mean()
    return out


def build_state(df: pd.DataFrame, window_seconds: float = 60.0) -> pd.DataFrame:
    """Flows (extended schema) -> timestamped window table with STATE_COLUMNS (+ metadata)."""
    base = build_windows(df, window_seconds=window_seconds)
    extra = _extra_features(df, window_seconds)
    state = base.merge(extra, left_on="window_id", right_index=True, how="left")
    cols = STATE_COLUMNS + [c for c in PACKET_ONLY_COLUMNS if c in state.columns]
    state[cols] = state[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return state


# ---------------------------------------------------------------- normalisation

def transform_raw(state: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """log1p the heavy-tailed columns; return float64 matrix in `cols` order."""
    x = state[cols].astype("float64").copy()
    for c in cols:
        if c in LOG_COLUMNS:
            x[c] = np.log1p(x[c].clip(lower=0))
    return x.to_numpy()


class StateScaler:
    """log1p (LOG_COLUMNS) then z-score. Fit on benign windows of the TRAINING part only."""

    def __init__(self, cols: list[str] | None = None):
        self.cols = list(cols or STATE_COLUMNS)
        self.mean_ = None
        self.std_ = None

    def fit(self, state: pd.DataFrame, mask=None) -> "StateScaler":
        x = transform_raw(state if mask is None else state[mask], self.cols)
        self.mean_ = x.mean(0)
        std = x.std(0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, state: pd.DataFrame) -> np.ndarray:
        return (transform_raw(state, self.cols) - self.mean_) / self.std_

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"cols": self.cols, "mean": self.mean_.tolist(), "std": self.std_.tolist(),
             "log_columns": LOG_COLUMNS}, indent=1))

    @classmethod
    def load(cls, path) -> "StateScaler":
        d = json.loads(Path(path).read_text())
        s = cls(d["cols"])
        s.mean_, s.std_ = np.array(d["mean"]), np.array(d["std"])
        return s


if __name__ == "__main__":
    import sys

    from data_loader import load_raw_csvs

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    ws = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    flows = load_raw_csvs(raw_dir, extended=True)
    state = build_state(flows, window_seconds=ws)
    del flows
    print(f"{len(state)} windows of {ws:g}s, {int(state['window_is_attack'].sum())} attack windows, "
          f"{len(STATE_COLUMNS)} state features")

    # Scaler fitted on benign windows of the first 60% of the timeline (a fixed, documented split;
    # the benchmark in Phase 3 refits per CV fold so nothing leaks).
    cut = int(len(state) * 0.6)
    fit_mask = (state.index < cut) & (state["window_is_attack"] == 0)
    scaler = StateScaler().fit(state, fit_mask)
    scaler.save("models/state_scaler.json")
    meta = ["window_id", "window_start", "window_end", "segment", "attack_fraction",
            "attack_type", "window_is_attack"]
    scaled = pd.concat([state[meta].reset_index(drop=True),
                        pd.DataFrame(scaler.transform(state), columns=STATE_COLUMNS)], axis=1)
    state[meta + STATE_COLUMNS].to_parquet("data/state_v1.parquet")
    scaled.to_parquet("data/state_v1_scaled.parquet")
    print("Saved data/state_v1.parquet, data/state_v1_scaled.parquet, models/state_scaler.json")

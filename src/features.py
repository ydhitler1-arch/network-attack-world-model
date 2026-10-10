"""
Bucket flows into fixed-width time windows and compute aggregate features per
window. This is the step that turns "a pile of flows" into "a time series we
can forecast over."

Each output row = one time window, with:
    window_start, window_end
    flow_count
    packet_rate           (packets / second)
    byte_rate              (bytes / second)
    unique_dst_ips
    unique_dst_ports
    syn_ratio               (SYN-heavy flows / total flows — scan/flood signal)
    src_ip_entropy          (Shannon entropy of source IPs — spoofing/DDoS signal)
    byte_ratio_asymmetry    (|fwd_bytes - bwd_bytes| / total — exfil signal)
    attack_fraction         (fraction of flows in window labeled as attack)
    window_is_attack        (1 if attack_fraction > threshold)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _top_share(s: pd.Series) -> float:
    vc = s.value_counts()
    return float(vc.iloc[0] / len(s))


def _entropy(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def build_windows(
    df: pd.DataFrame,
    window_seconds: float = 10.0,
    attack_threshold: float = 0.05,
    max_gap_windows: int = 30,
) -> pd.DataFrame:
    """Aggregate flow-level records into fixed-width time windows."""
    t0 = df["timestamp"].min()
    df = df.copy()
    df["window_id"] = ((df["timestamp"] - t0) // window_seconds).astype(int)

    rows = []
    for wid, g in df.groupby("window_id", sort=True):
        total_packets = g["fwd_packets"].sum() + g["bwd_packets"].sum()
        total_bytes = g["fwd_bytes"].sum() + g["bwd_bytes"].sum()
        fwd_bytes = g["fwd_bytes"].sum()
        bwd_bytes = g["bwd_bytes"].sum()

        rows.append(
            {
                "window_id": wid,
                "window_start": t0 + wid * window_seconds,
                "window_end": t0 + (wid + 1) * window_seconds,
                "flow_count": len(g),
                "packet_rate": total_packets / window_seconds,
                "byte_rate": total_bytes / window_seconds,
                "unique_dst_ips": g["dst_ip"].nunique(),
                "unique_dst_ports": g["dst_port"].nunique(),
                "syn_ratio": (g["syn_flag"] > 0).mean(),
                "src_ip_entropy": _entropy(g["src_ip"].value_counts()),
                "byte_ratio_asymmetry": (
                    abs(fwd_bytes - bwd_bytes) / total_bytes if total_bytes > 0 else 0.0
                ),
                # Concentration features: floods, scans and brute force all look like one
                # (or few) sources hammering one target, which averages and counts miss.
                "unique_src_ips": g["src_ip"].nunique(),
                "top_src_share": _top_share(g["src_ip"]),
                "top_dstport_share": _top_share(g["dst_port"]),
                "top_pair_share": float(
                    g.groupby(["src_ip", "dst_ip", "dst_port"]).size().max() / len(g)
                ),
                "mean_log_duration": float(np.log1p(g["duration"].clip(lower=0)).mean()),
                "attack_fraction": g["is_attack"].mean(),
                "attack_type": (
                    g.loc[g["is_attack"] == 1, "label"].mode().iat[0]
                    if g["is_attack"].any() else ""
                ),
            }
        )

    windows = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)

    # Capture days are separated by long idle gaps (nights). Never let a sequence span
    # one: split into segments at gaps > max_gap_windows, and zero-fill the short gaps
    # inside a segment (a minute with no flows is real information: zero traffic).
    gap = windows["window_id"].diff().fillna(1) > max_gap_windows
    windows["segment"] = gap.cumsum().astype(int)
    parts = []
    for seg, g in windows.groupby("segment"):
        g = g.set_index("window_id").reindex(range(int(g["window_id"].min()), int(g["window_id"].max()) + 1))
        g["segment"] = seg
        g.index.name = "window_id"
        parts.append(g.reset_index())
    windows = pd.concat(parts, ignore_index=True)
    skip = ("window_start", "window_end", "segment", "window_id", "attack_type")
    fill_cols = [c for c in windows.columns if c not in skip]
    windows[fill_cols] = windows[fill_cols].fillna(0)
    windows["attack_type"] = windows["attack_type"].fillna("")
    windows["window_start"] = t0 + windows["window_id"] * window_seconds
    windows["window_end"] = windows["window_start"] + window_seconds
    windows["window_is_attack"] = (windows["attack_fraction"] > attack_threshold).astype(int)
    windows.loc[windows["window_is_attack"] == 0, "attack_type"] = ""
    return windows


FEATURE_COLUMNS = [
    "flow_count",
    "packet_rate",
    "byte_rate",
    "unique_dst_ips",
    "unique_dst_ports",
    "syn_ratio",
    "src_ip_entropy",
    "byte_ratio_asymmetry",
    "unique_src_ips",
    "top_src_share",
    "top_dstport_share",
    "top_pair_share",
    "mean_log_duration",
]


if __name__ == "__main__":
    import sys

    from data_loader import load_raw_csvs

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_raw_csvs(raw_dir)
    # CICIDS2017 timestamps are minute-resolution -> use 60s windows for it.
    window_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    windows = build_windows(df, window_seconds=window_seconds)
    print(f"{len(df):,} flows -> {len(windows)} windows of {window_seconds:g}s, "
          f"{int(windows['window_is_attack'].sum())} attack windows")
    print(windows.shape)
    print(windows[["window_id", "packet_rate", "syn_ratio", "window_is_attack"]].head(20))
    windows.to_parquet("data/windows.parquet")
    print("Saved to data/windows.parquet")

"""
Generates a small synthetic flow dataset so you can run the entire pipeline
(features -> sequences -> training -> dashboard) immediately, without waiting
on a multi-GB dataset download. Swap in real CICIDS2017 CSVs later — same
downstream code works unchanged since it conforms to the data_loader schema.

Simulates: a quiet baseline, punctuated by a few synthetic "attack campaigns"
(a port-scan-like ramp-up followed by a volumetric burst) so there's an actual
precursor signal for the model to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def _benign_flows(n: int, t_start: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": t_start + np.sort(rng.uniform(0, 1, n)),
            "src_ip": [f"10.0.0.{rng.integers(2, 60)}" for _ in range(n)],
            "dst_ip": [f"10.0.1.{rng.integers(2, 20)}" for _ in range(n)],
            "dst_port": rng.choice([80, 443, 22, 53], n),
            "protocol": "6",
            "duration": rng.uniform(0.01, 2.0, n),
            "fwd_packets": rng.integers(1, 20, n),
            "bwd_packets": rng.integers(1, 20, n),
            "fwd_bytes": rng.integers(40, 1500, n),
            "bwd_bytes": rng.integers(40, 1500, n),
            "syn_flag": rng.integers(0, 2, n),
            "label": "BENIGN",
        }
    )


def _recon_ramp_flows(n: int, t_start: float) -> pd.DataFrame:
    """Precursor phase: single source, fanning out across many dst ports — a scan."""
    src = f"10.0.0.{rng.integers(60, 70)}"
    return pd.DataFrame(
        {
            "timestamp": t_start + np.sort(rng.uniform(0, 1, n)),
            "src_ip": src,
            "dst_ip": [f"10.0.1.{rng.integers(2, 254)}" for _ in range(n)],
            "dst_port": rng.integers(1, 65535, n),
            "protocol": "6",
            "duration": rng.uniform(0.001, 0.05, n),
            "fwd_packets": 1,
            "bwd_packets": 0,
            "fwd_bytes": 40,
            "bwd_bytes": 0,
            "syn_flag": 1,
            "label": "BENIGN",  # recon itself is subtle/unlabeled — that's the point
        }
    )


def _volumetric_attack_flows(n: int, t_start: float) -> pd.DataFrame:
    """The actual labeled attack: SYN-flood-like burst from spoofed sources."""
    return pd.DataFrame(
        {
            "timestamp": t_start + np.sort(rng.uniform(0, 1, n)),
            "src_ip": [f"192.168.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(n)],
            "dst_ip": "10.0.1.5",
            "dst_port": 80,
            "protocol": "6",
            "duration": rng.uniform(0.001, 0.01, n),
            "fwd_packets": 1,
            "bwd_packets": 0,
            "fwd_bytes": 60,
            "bwd_bytes": 0,
            "syn_flag": 1,
            "label": "DDoS",
        }
    )


def generate(
    total_seconds: int = 1800,
    out_path: str = "data/raw/synthetic.csv",
    n_campaigns: int = 4,
) -> pd.DataFrame:
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []
    t = 0.0
    # Plant 4 campaigns: ~90s recon ramp, then ~20s attack burst, spaced through the run.
    # Evenly spaced slots with jitter so campaigns never overlap, however many there are.
    slot = (total_seconds - 400) / n_campaigns
    assert slot > 150, "total_seconds too small for n_campaigns"
    campaign_starts = [200 + i * slot + rng.uniform(0, slot - 130) for i in range(n_campaigns)]

    while t < total_seconds:
        in_campaign = any(cs <= t < cs + 110 for cs in campaign_starts)
        if in_campaign:
            cs = next(cs for cs in campaign_starts if cs <= t < cs + 110)
            phase_t = t - cs
            if phase_t < 90:
                frames.append(_recon_ramp_flows(rng.integers(20, 40), t))
            else:
                frames.append(_volumetric_attack_flows(rng.integers(200, 400), t))
        else:
            frames.append(_benign_flows(rng.integers(30, 60), t))
        t += 1.0

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic flows to {out_path}")
    print(f"Campaign start times (s): {[round(c) for c in campaign_starts]}")
    return df


if __name__ == "__main__":
    import sys

    # usage: python make_synthetic_data.py [total_seconds] [n_campaigns]
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    generate(total_seconds=total, n_campaigns=n)

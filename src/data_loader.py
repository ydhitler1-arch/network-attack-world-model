"""
Load a public labeled network-flow dataset and normalize it into a common schema.

Supports CICIDS2017/2018-style CSVs out of the box. The goal is just to get to a
DataFrame with these columns, regardless of source dataset quirks:

    timestamp   : float, seconds (relative or absolute — only ordering matters)
    src_ip      : str
    dst_ip      : str
    dst_port    : int
    protocol    : str
    duration    : float, flow duration in seconds
    fwd_packets : int
    bwd_packets : int
    fwd_bytes   : int
    bwd_bytes   : int
    syn_flag    : int, count of SYN flags in the flow
    label       : str, 'BENIGN' or an attack name (dataset-specific)

CICIDS column names vary by release (extra spaces, different casing), so this
loader is deliberately tolerant and maps by fuzzy-matching known aliases.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

# Known column aliases across CICIDS2017/2018 CSV releases.
COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["Timestamp"],
    "src_ip": ["Src IP", "Source IP"],
    "dst_ip": ["Dst IP", "Destination IP"],
    "dst_port": ["Dst Port", "Destination Port"],
    "protocol": ["Protocol"],
    "duration": ["Flow Duration"],
    "fwd_packets": ["Tot Fwd Pkts", "Total Fwd Packets"],
    "bwd_packets": ["Tot Bwd Pkts", "Total Backward Packets"],
    "fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets"],
    "bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets"],
    "syn_flag": ["SYN Flag Cnt", "SYN Flag Count"],
    "label": ["Label"],
}


# Extra per-flow columns used by the world-model state vector (src/state.py). Loaded only
# with extended=True so the original pipeline (features.py, evaluate.py) is unchanged.
EXTENDED_ALIASES: dict[str, list[str]] = {
    "src_port": ["Src Port", "Source Port"],
    "fin_flag": ["FIN Flag Cnt", "FIN Flag Count"],
    "rst_flag": ["RST Flag Cnt", "RST Flag Count"],
    "psh_flag": ["PSH Flag Cnt", "PSH Flag Count"],
    "ack_flag": ["ACK Flag Cnt", "ACK Flag Count"],
    "urg_flag": ["URG Flag Cnt", "URG Flag Count"],
    "flow_iat_mean": ["Flow IAT Mean"],
    "flow_iat_std": ["Flow IAT Std"],
    "flow_iat_max": ["Flow IAT Max"],
    "fwd_iat_mean": ["Fwd IAT Mean"],
    "bwd_iat_mean": ["Bwd IAT Mean"],
    "avg_pkt_size": ["Average Packet Size", "Pkt Size Avg"],
    "pkt_len_std": ["Packet Length Std", "Pkt Len Std"],
    "pkt_len_var": ["Packet Length Variance", "Pkt Len Var"],
    "init_win_fwd": ["Init_Win_bytes_forward", "Init Fwd Win Byts"],
    "init_win_bwd": ["Init_Win_bytes_backward", "Init Bwd Win Byts"],
    "down_up_ratio": ["Down/Up Ratio"],
}


def _normalize_colname(c: str) -> str:
    return c.strip()


def _build_rename_map(columns: list[str], alias_map: dict[str, list[str]] = COLUMN_ALIASES) -> dict[str, str]:
    cleaned = {_normalize_colname(c): c for c in columns}
    rename = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in cleaned:
                rename[cleaned[alias]] = canonical
                break
    return rename


def load_raw_csvs(raw_dir: str | Path, extended: bool = False) -> pd.DataFrame:
    """Load and concatenate every CSV in raw_dir, normalizing columns as we go.

    extended=True also loads the EXTENDED_ALIASES columns (flags, IAT, packet size, TCP window).
    """
    raw_dir = Path(raw_dir)
    alias_map = {**COLUMN_ALIASES, **EXTENDED_ALIASES} if extended else COLUMN_ALIASES
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in {raw_dir}. Download CICIDS2017/2018 (or UNSW-NB15) "
            f"CSVs there first — see README.md."
        )

    frames = [_read_one(f, alias_map) for f in files]
    full = pd.concat(frames, ignore_index=True)
    full = _coerce_types(full)
    return full


def _read_one(f, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    # cp1252: the CICIDS2017 Thursday file has a non-UTF-8 en dash in its labels.
    # usecols: load only the columns we map (the files have ~85, we need 12).
    wanted = {a for al in alias_map.values() for a in al} | set(alias_map)  # + canonical names (synthetic data)
    df = pd.read_csv(
        f, low_memory=False, encoding="cp1252",
        usecols=lambda c: c.strip() in wanted,
    )
    rename = _build_rename_map(list(df.columns), alias_map)
    df = df.rename(columns=rename)
    missing = set(alias_map) - set(df.columns)
    if missing:
        # Not fatal — some dataset variants omit a field like syn_flag.
        for col in missing:
            df[col] = "UNLABELLED" if col == "label" else 0
    # Parse timestamps per file: formats differ between files (e.g. Monday has seconds),
    # and a single to_datetime over the concatenation would coerce the odd ones to NaT.
    # CICIDS2017 dates are day-first (3/7/2017 = 3 July).
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], dayfirst=True, errors="coerce", format="mixed"
        )
    return df[list(alias_map.keys())]


def load_csv_file(path: str | Path, extended: bool = True) -> pd.DataFrame:
    """Load ONE flow CSV (CICFlowMeter layout, or the canonical-column layout the demo sample uses)."""
    alias_map = {**COLUMN_ALIASES, **EXTENDED_ALIASES} if extended else COLUMN_ALIASES
    return _coerce_types(_read_one(path, alias_map))


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "dst_port",
        "duration",
        "fwd_packets",
        "bwd_packets",
        "fwd_bytes",
        "bwd_bytes",
        "syn_flag",
    ]
    numeric_cols += [c for c in EXTENDED_ALIASES if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if c in EXTENDED_ALIASES:
            df[c] = df[c].replace([float("inf"), float("-inf")], 0).astype("float32")

    df["label"] = df["label"].astype(str).str.strip()
    df["is_attack"] = (~df["label"].str.upper().isin(["BENIGN", "UNLABELLED"])).astype(int)

    # Timestamp: CICIDS stores it as a string like "3/7/2017 9:51". Our synthetic
    # generator instead writes plain float seconds — if the column is already
    # numeric, treat it as seconds directly (pd.to_datetime would otherwise
    # misread small floats as nanosecond epoch offsets, collapsing everything
    # near 1970-01-01).
    raw_ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(raw_ts):
        df["timestamp"] = pd.to_numeric(raw_ts, errors="coerce").fillna(0.0)
    else:
        ts = pd.to_datetime(raw_ts, errors="coerce")
        if ts.isna().all():
            df["timestamp"] = df.index.astype(float)
        else:
            # CICIDS2017 timestamps have no AM/PM marker and the capture ran ~8am-5pm,
            # so hours 1-7 are really afternoon (13-19). Without this fix, afternoon
            # flows sort before morning flows and the timeline is scrambled.
            pm = ts.dt.hour.between(1, 7)
            ts = ts.where(~pm, ts + pd.Timedelta(hours=12))
            df["timestamp"] = ts.astype("datetime64[ns]").astype("int64") / 1e9  # seconds since epoch
            # Blank/garbled rows exist in the CICIDS CSVs; they have no valid time, so drop
            # them (forward-filling would invent a fake 1970-era window).
            valid = (ts.notna() & (ts.dt.year >= 2000)).to_numpy()
            print(f"  dropping {(~valid).sum():,} rows with invalid timestamps")
            df = df[valid].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_raw_csvs(raw_dir)
    print(df.shape)
    print(df["label"].value_counts().head(20))

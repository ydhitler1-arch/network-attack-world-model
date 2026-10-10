"""
PCAP -> flow records (optional path, needs Scapy: pip install scapy).

Produces a DataFrame in the same schema as data_loader.load_raw_csvs(extended=True), plus
packet-level columns a CSV cannot provide:

    ttl_mean, ttl_var     TTL of all packets in the flow, mean and variance
    tcp_win_mean          mean TCP window field
    df_count, frag_count  packets with Don't-Fragment set / with MF or a fragment offset
    retrans_count         TCP data segments whose (direction, seq, payload length) was seen before
    payload_std           std of per-packet payload length (same as pkt_len_std)

Flow definition (simplified CICFlowMeter): bidirectional 5-tuple, the first packet's sender is
"forward", a flow ends after `flow_timeout` seconds without packets (default 120 s, as in
CICFlowMeter) or at the end of the file. Unlike CICFlowMeter it does not end flows at FIN/RST,
so a reused 5-tuple within the timeout is merged into one flow. Duration and IAT are in
microseconds to match the CSVs. IPv4 only (IPv6 packets are counted and skipped).
Labels are unknown: label = "UNLABELLED", is_attack = 0 (inference only).

Speed: Scapy parses roughly 10-30k packets/s, so a 1 GB PCAP takes minutes; use max_packets
to cap it.

    python src/pcap_to_flows.py capture.pcap out_flows.parquet
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from scapy.all import IP, TCP, UDP, PcapReader  # noqa: F401
    _SCAPY_ERR = None
except ImportError as e:  # pragma: no cover
    _SCAPY_ERR = e


class _Acc:
    """Running mean/variance/max (Welford) without storing samples."""
    __slots__ = ("n", "mean", "m2", "max")

    def __init__(self):
        self.n, self.mean, self.m2, self.max = 0, 0.0, 0.0, 0.0

    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if x > self.max:
            self.max = x

    @property
    def var(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


class _Flow:
    __slots__ = ("start", "last", "last_dir", "src", "dst", "sport", "dport", "proto",
                 "pk", "by", "flags", "iat", "fiat", "biat", "tlast", "ttl", "win", "plen",
                 "init_win", "df", "frag", "retx", "seen", "pk_hdr")

    def __init__(self, t, src, dst, sport, dport, proto):
        self.start = self.last = t
        self.src, self.dst, self.sport, self.dport, self.proto = src, dst, sport, dport, proto
        self.pk = [0, 0]
        self.by = [0, 0]
        self.flags = {"F": 0, "S": 0, "R": 0, "P": 0, "A": 0, "U": 0}
        self.iat, self.fiat, self.biat = _Acc(), _Acc(), _Acc()
        self.tlast = [None, None]
        self.ttl, self.win, self.plen = _Acc(), _Acc(), _Acc()
        self.init_win = [-1, -1]
        self.df = self.frag = self.retx = 0
        self.seen = set()


def _flush(f: _Flow) -> dict:
    tp = f.pk[0] + f.pk[1]
    tb = f.by[0] + f.by[1]
    return {
        "timestamp": f.start, "src_ip": f.src, "dst_ip": f.dst, "src_port": f.sport,
        "dst_port": f.dport, "protocol": f.proto,
        "duration": (f.last - f.start) * 1e6,
        "fwd_packets": f.pk[0], "bwd_packets": f.pk[1], "fwd_bytes": f.by[0], "bwd_bytes": f.by[1],
        "syn_flag": f.flags["S"], "fin_flag": f.flags["F"], "rst_flag": f.flags["R"],
        "psh_flag": f.flags["P"], "ack_flag": f.flags["A"], "urg_flag": f.flags["U"],
        "flow_iat_mean": f.iat.mean * 1e6, "flow_iat_std": f.iat.std * 1e6, "flow_iat_max": f.iat.max * 1e6,
        "fwd_iat_mean": f.fiat.mean * 1e6, "bwd_iat_mean": f.biat.mean * 1e6,
        "avg_pkt_size": tb / tp if tp else 0.0,
        "pkt_len_std": f.plen.std, "pkt_len_var": f.plen.var,
        "init_win_fwd": f.init_win[0], "init_win_bwd": f.init_win[1],
        "down_up_ratio": f.pk[1] / f.pk[0] if f.pk[0] else 0.0,
        "label": "UNLABELLED", "is_attack": 0,
        # packet-level (PCAP only)
        "ttl_mean": f.ttl.mean, "ttl_var": f.ttl.var, "tcp_win_mean": f.win.mean,
        "df_count": f.df, "frag_count": f.frag, "retrans_count": f.retx, "payload_std": f.plen.std,
    }


def pcap_to_flows(path: str, flow_timeout: float = 120.0, max_packets: int | None = None) -> pd.DataFrame:
    if _SCAPY_ERR is not None:
        raise ImportError("Scapy is required for PCAP input: pip install scapy") from _SCAPY_ERR
    flows: dict[tuple, _Flow] = {}
    done: list[dict] = []
    n_pkts = n_skipped = 0
    last_sweep = None

    with PcapReader(path) as reader:
        for pkt in reader:
            if max_packets and n_pkts >= max_packets:
                break
            n_pkts += 1
            if IP not in pkt:
                n_skipped += 1
                continue
            ip = pkt[IP]
            t = float(pkt.time)
            if TCP in pkt:
                l4, proto, sport, dport = pkt[TCP], 6, pkt[TCP].sport, pkt[TCP].dport
                payload = len(l4.payload)
            elif UDP in pkt:
                l4, proto, sport, dport = pkt[UDP], 17, pkt[UDP].sport, pkt[UDP].dport
                payload = len(l4.payload)
            else:
                l4, proto, sport, dport = None, int(ip.proto), 0, 0
                payload = len(ip.payload)

            a, b = (ip.src, sport), (ip.dst, dport)
            key = (proto,) + ((a, b) if a <= b else (b, a))
            f = flows.get(key)
            if f is not None and t - f.last > flow_timeout:
                done.append(_flush(flows.pop(key)))
                f = None
            if f is None:
                f = flows[key] = _Flow(t, ip.src, ip.dst, sport, dport, proto)
            d = 0 if (ip.src, sport) == (f.src, f.sport) else 1

            if f.iat.n or f.pk[0] + f.pk[1]:
                f.iat.add(t - f.last)
            if f.tlast[d] is not None:
                (f.fiat if d == 0 else f.biat).add(t - f.tlast[d])
            f.tlast[d] = t
            f.last = t
            f.pk[d] += 1
            f.by[d] += payload
            f.plen.add(payload)
            f.ttl.add(int(ip.ttl))
            if int(ip.flags) & 0x2:  # DF
                f.df += 1
            if (int(ip.flags) & 0x1) or int(ip.frag) > 0:  # MF or fragment offset
                f.frag += 1
            if proto == 6:
                fl = str(l4.flags)
                for c in f.flags:
                    if c in fl:
                        f.flags[c] += 1
                f.win.add(int(l4.window))
                if f.init_win[d] < 0:
                    f.init_win[d] = int(l4.window)
                if payload > 0:
                    sig = (d, int(l4.seq), payload)
                    if sig in f.seen:
                        f.retx += 1
                    elif len(f.seen) < 100_000:
                        f.seen.add(sig)

            if last_sweep is None:
                last_sweep = t
            elif t - last_sweep > flow_timeout:  # bound memory on long captures
                for k in [k for k, v in flows.items() if t - v.last > flow_timeout]:
                    done.append(_flush(flows.pop(k)))
                last_sweep = t

    done += [_flush(f) for f in flows.values()]
    df = pd.DataFrame(done)
    if df.empty:
        raise ValueError(f"No IPv4 flows found in {path} ({n_pkts} packets, {n_skipped} non-IPv4)")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["packets"], df.attrs["skipped_non_ipv4"] = n_pkts, n_skipped
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in ("timestamp",):
            df[c] = df[c].astype("float32")
    return df


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/pcap_flows.parquet"
    out = pcap_to_flows(src)
    out.to_parquet(dst)
    print(f"{out.attrs['packets']:,} packets -> {len(out):,} flows "
          f"({out.attrs['skipped_non_ipv4']:,} non-IPv4 skipped). Saved {dst}")

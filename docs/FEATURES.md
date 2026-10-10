# State vector S_t (Phase 1)

One state = one 60 s window of flows. 43 core features computed by `src/state.py` from CIC-IDS2017 CSVs (or from PCAP-derived flows via `src/pcap_to_flows.py`). Raw aggregates are in `data/state_v1.parquet`; the normalised matrix (log1p on heavy-tailed columns, then z-score) is in `data/state_v1_scaled.parquet`; scaler parameters are in `models/state_scaler.json`.

Normalisation in the shipped file uses benign windows of the first 60% of the timeline. The Phase 3 benchmark refits the scaler inside each CV fold so nothing leaks.

## Core features (work for both CSV and PCAP input)

**Base (13, from `src/features.py`, unchanged; verified identical to `data/windows.parquet`):**
flow_count, packet_rate, byte_rate, unique_dst_ips, unique_dst_ports, syn_ratio, src_ip_entropy, byte_ratio_asymmetry, unique_src_ips, top_src_share, top_dstport_share, top_pair_share, mean_log_duration.

**Flow level (22):**

| requirement item | feature(s) |
|---|---|
| TCP flag counts | fin_ratio, rst_ratio, psh_ratio, ack_ratio, urg_ratio (share of flows with that flag), syn_only_ratio (SYN and no ACK: half-open/scan) , plus syn_ratio from the base set |
| protocol | tcp_share, udp_share |
| bytes/packets per flow | mean_pkts_per_flow, mean_bytes_per_flow |
| IAT stats (mean/variance/max) | log_iat_mean, log_iat_std, log_iat_max (mean over flows of log1p of the per-flow IAT mean / std / max) |
| bidirectional ratios | pkt_ratio_bwd_fwd, byte_ratio_bwd_fwd, mean_down_up_ratio |
| scan / flood signatures | no_reply_ratio (no backward packet), single_pkt_ratio (<= 2 packets), max_ports_per_src (vertical scan), max_dsts_per_src (horizontal scan), dst_port_entropy, src_port_uniq_ratio |
| src/dst IP + port | via the concentration and entropy features above (individual addresses are not model inputs) |

**Packet-level proxies available from the CSV (8):**
mean_log_init_win_fwd, std_log_init_win_fwd, win_missing_ratio (TCP window of the first forward packet), mean_avg_pkt_size, std_avg_pkt_size, small_pkt_ratio (< 100 B), large_pkt_ratio (>= 1000 B) (payload-size distribution at flow granularity), mean_pkt_len_std (within-flow packet-size spread).

## What a CSV dataset cannot provide

| packet-level feature | in CIC-IDS2017 CSV? | note |
|---|---|---|
| TTL and TTL variance | **No** | needs packets |
| IP fragmentation flags (DF/MF, offset) | **No** | needs packets |
| Retransmissions | **No** | needs TCP sequence numbers |
| Per-packet TCP window over time | **No** | only the first-packet window per direction (`Init_Win_bytes_*`) |
| True payload-size distribution | **Partly** | per-flow average, std, min/max only, no per-packet histogram |
| Port-scan signature | **Derived** | from flow aggregates (ports per source, no-reply share), not from packet timing |

## PCAP-only features (`PACKET_ONLY_COLUMNS`)

`src/pcap_to_flows.py` (needs `pip install scapy`) builds flow records from a PCAP and adds: mean_ttl, ttl_var_mean, tcp_win_mean, frag_ratio, retrans_ratio, payload_std_mean (window-level aggregates of the per-flow columns ttl_mean, ttl_var, tcp_win_mean, frag_count, retrans_count, payload_std). They appear in `build_state` output when the flows come from a PCAP. They are **not** in the core 43 because the world model must also run on CSV input, and the CIC-IDS2017 training data cannot supply them. Treat them as extra context for the demo and for future PCAP-trained models.

## Known limitations

- CIC-IDS2017 timestamps are minute-resolution, so 60 s is the finest meaningful window.
- `pcap_to_flows` is a simplified CICFlowMeter: idle-timeout (120 s) flow splitting only, no FIN/RST termination, IPv4 only, Scapy speed (about 10-30k packets/s). Its TCP-flag, IAT and duration definitions follow CICFlowMeter but were not validated against CICFlowMeter output on the same PCAP (the 2017 PCAPs are not in the repo).
- CICFlowMeter 2017 has known quirks (for example URG counts and negative IAT values). Negative IATs are clipped to 0.
- `retrans_count` counts repeated (direction, seq, payload length) data segments, an approximation of retransmission.
- Median differences between attack and benign windows are small for most new features (for example median `ack_ratio` 0.30 vs 0.35); only `max_ports_per_src` clearly moves (11 vs 27). Whether the new features help is decided by the Phase 3 benchmark, not assumed here.

## Reproduce

```bash
python src/state.py data/raw 60
python -W ignore tests/test_phase1.py
```

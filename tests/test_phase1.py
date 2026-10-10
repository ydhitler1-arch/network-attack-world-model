"""Phase 1 checks: PCAP -> flows -> state vector, using a small synthetic PCAP built with Scapy.

    python tests/test_phase1.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scapy.all import DNS, DNSQR, IP, TCP, UDP, Ether, wrpcap  # noqa: E402

from data_loader import COLUMN_ALIASES, EXTENDED_ALIASES  # noqa: E402
from pcap_to_flows import pcap_to_flows  # noqa: E402
from state import PACKET_ONLY_COLUMNS, STATE_COLUMNS, StateScaler, build_state  # noqa: E402

T0 = 1_700_000_000.0
pkts = []


def add(p, t):
    p.time = T0 + t
    pkts.append(p)


C, S, A = "10.0.0.5", "10.0.0.9", "10.0.0.66"

# 1) normal TCP session with one retransmitted data segment, varying TTL on the reply path
add(Ether() / IP(src=C, dst=S, ttl=64) / TCP(sport=40000, dport=80, flags="S", seq=100, window=64240), 0.00)
add(Ether() / IP(src=S, dst=C, ttl=128) / TCP(sport=80, dport=40000, flags="SA", seq=500, ack=101, window=8192), 0.01)
add(Ether() / IP(src=C, dst=S, ttl=64) / TCP(sport=40000, dport=80, flags="A", seq=101, ack=501, window=64240), 0.02)
data = Ether() / IP(src=C, dst=S, ttl=64, flags="DF") / TCP(sport=40000, dport=80, flags="PA", seq=101, ack=501, window=64240) / ("x" * 500)
add(data, 0.03)
add(data.copy(), 0.30)  # retransmission: same seq, same payload length
add(Ether() / IP(src=S, dst=C, ttl=120) / TCP(sport=80, dport=40000, flags="A", seq=501, ack=601, window=8192), 0.31)
add(Ether() / IP(src=C, dst=S, ttl=64) / TCP(sport=40000, dport=80, flags="FA", seq=601, ack=501, window=64240), 0.40)

# 2) SYN scan: 30 ports, only some answered with RST
for i, port in enumerate(range(1000, 1030)):
    add(Ether() / IP(src=A, dst=S, ttl=47) / TCP(sport=50000 + i, dport=port, flags="S", seq=1, window=1024), 1.0 + i * 0.01)
    if i % 3 == 0:
        add(Ether() / IP(src=S, dst=A, ttl=64) / TCP(sport=port, dport=50000 + i, flags="RA", seq=0, ack=2, window=0), 1.005 + i * 0.01)

# 3) UDP DNS exchange
add(Ether() / IP(src=C, dst="10.0.0.53", ttl=64) / UDP(sport=5353, dport=53) / DNS(rd=1, qd=DNSQR(qname="example.com")), 2.0)
add(Ether() / IP(src="10.0.0.53", dst=C, ttl=64) / UDP(sport=53, dport=5353) / DNS(qr=1, qd=DNSQR(qname="example.com")), 2.02)

# 4) fragmented packet (MF set)
add(Ether() / IP(src=A, dst=S, ttl=47, flags="MF", frag=0, proto=17) / ("y" * 64), 3.0)

tmp = os.path.join(tempfile.mkdtemp(), "t.pcap")
wrpcap(tmp, pkts)
fl = pcap_to_flows(tmp)

# ---- flow extraction
tcp = fl[(fl.src_port == 40000) & (fl.dst_port == 80)].iloc[0]
assert tcp.fwd_packets == 5 and tcp.bwd_packets == 2, (tcp.fwd_packets, tcp.bwd_packets)
assert tcp.fwd_bytes == 1000 and tcp.bwd_bytes == 0, tcp.fwd_bytes   # two 500-byte segments (one retransmitted)
assert tcp.retrans_count == 1, tcp.retrans_count
assert tcp.df_count == 2 and tcp.frag_count == 0
ttls = [64, 128, 64, 64, 64, 120, 64]
assert np.isclose(tcp.ttl_mean, np.mean(ttls), atol=1e-3) and np.isclose(tcp.ttl_var, np.var(ttls), atol=1e-2), (tcp.ttl_mean, tcp.ttl_var)
assert tcp.syn_flag == 2 and tcp.fin_flag == 1 and tcp.psh_flag == 2
assert tcp.init_win_fwd == 64240 and tcp.init_win_bwd == 8192
assert abs(tcp.duration - 0.40e6) < 1, tcp.duration
scan = fl[(fl.src_ip == A) & fl.dst_port.between(1000, 1029)]
assert len(scan) == 30 and (scan.bwd_packets == 0).sum() == 20, (len(scan), (scan.bwd_packets == 0).sum())
assert (fl.frag_count > 0).sum() == 1
assert ((fl.protocol == 17) & (fl.dst_port == 53)).sum() == 1

# ---- schema parity with the CSV loader (extended): everything the state builder needs exists
need = set(COLUMN_ALIASES) | set(EXTENDED_ALIASES) | {"is_attack"}
assert need <= set(fl.columns), need - set(fl.columns)

# ---- state vector from PCAP flows
st = build_state(fl, window_seconds=10.0)
assert set(STATE_COLUMNS) <= set(st.columns) and set(PACKET_ONLY_COLUMNS) <= set(st.columns)
row = st.iloc[0]
assert row.max_ports_per_src >= 30, row.max_ports_per_src
assert row.no_reply_ratio > 0.5 and row.syn_only_ratio > 0.5
assert row.retrans_ratio > 0 and row.frag_ratio > 0
mat = StateScaler().fit(st).transform(st)
assert np.isfinite(mat).all() and mat.shape == (len(st), len(STATE_COLUMNS))
print("OK - %d flows from %d packets; scan window: ports/src=%d, no_reply=%.2f, ttl_var_mean=%.2f"
      % (len(fl), fl.attrs["packets"], row.max_ports_per_src, row.no_reply_ratio, row.ttl_var_mean))

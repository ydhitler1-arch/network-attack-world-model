"""
Build the two small files the offline demo ships with (data/sample/):

  demo_flows.csv.gz   ~55 minutes of REAL CIC-IDS2017 Friday-afternoon flows (benign lead-in, then the
                      DDoS attack), in the canonical column layout the loader accepts. Labels are kept so
                      the demo can overlay ground truth.
  demo_traffic.pcap   12 minutes of SYNTHETIC traffic (benign web/DNS sessions, a SYN port scan in
                      minutes 8-9, an SSH-style connection burst in minute 10). It exists to exercise the
                      PCAP path end to end; it is not CIC-IDS2017 traffic, so the model's scores on it
                      say little about real-world performance.

    python src/make_demo_data.py [raw_dir]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("data/sample")


def make_csv(raw_dir: str):
    from data_loader import load_raw_csvs
    flows = load_raw_csvs(raw_dir, extended=True)
    ts = pd.to_datetime(flows["timestamp"], unit="s")
    day = ts.dt.strftime("%Y-%m-%d") == "2017-07-07"
    sl = flows[day & (ts.dt.hour * 60 + ts.dt.minute).between(15 * 60 + 35, 16 * 60 + 30)].copy()
    sl["timestamp"] = sl["timestamp"].astype("float64")
    OUT.mkdir(parents=True, exist_ok=True)
    sl.to_csv(OUT / "demo_flows.csv.gz", index=False, compression="gzip")
    print(f"demo_flows.csv.gz: {len(sl):,} flows, labels {sl['label'].value_counts().to_dict()}")


def make_pcap():
    from scapy.all import IP, TCP, UDP, Ether, wrpcap
    rnd = random.Random(7)
    T0 = 1_700_000_000.0
    pk = []

    def add(p, t):
        p.time = T0 + t
        pk.append(p)

    clients = [f"10.1.0.{i}" for i in range(10, 40)]
    servers = [("10.1.1.5", 80), ("10.1.1.6", 443), ("10.1.1.7", 80), ("10.1.1.8", 8080), ("10.1.1.9", 25)]
    seq = 1000

    def session(t, c, s, dport, sport, nbytes=400, ttl_c=64, ttl_s=64):
        nonlocal seq
        add(Ether() / IP(src=c, dst=s, ttl=ttl_c) / TCP(sport=sport, dport=dport, flags="S", seq=seq, window=64240), t)
        add(Ether() / IP(src=s, dst=c, ttl=ttl_s) / TCP(sport=dport, dport=sport, flags="SA", seq=seq + 7, ack=seq + 1, window=29200), t + 0.01)
        add(Ether() / IP(src=c, dst=s, ttl=ttl_c) / TCP(sport=sport, dport=dport, flags="A", seq=seq + 1, ack=seq + 8, window=64240), t + 0.02)
        tt = t + 0.03
        for _ in range(rnd.randint(1, 5)):
            add(Ether() / IP(src=c, dst=s, ttl=ttl_c, flags="DF") / TCP(sport=sport, dport=dport, flags="PA", seq=seq + 1, ack=seq + 8, window=64240) / ("q" * rnd.randint(60, 300)), tt)
            add(Ether() / IP(src=s, dst=c, ttl=ttl_s, flags="DF") / TCP(sport=dport, dport=sport, flags="PA", seq=seq + 8, ack=seq + 2, window=29200) / ("r" * rnd.randint(min(200, nbytes * 2), nbytes * 3)), tt + 0.02)
            tt += rnd.uniform(0.05, 0.6)
        add(Ether() / IP(src=c, dst=s, ttl=ttl_c) / TCP(sport=sport, dport=dport, flags="FA", seq=seq + 3, ack=seq + 9, window=64240), tt)
        seq += 5000

    # benign: ~4 sessions/s for 12 minutes, plus DNS
    t = 0.0
    while t < 720:
        c = rnd.choice(clients)
        s, dp = rnd.choice(servers)
        session(t, c, s, dp, rnd.randint(32768, 60999), ttl_c=rnd.choice([64, 64, 128]), ttl_s=rnd.choice([64, 128]))
        if rnd.random() < 0.5:
            add(Ether() / IP(src=c, dst="10.1.1.53", ttl=64) / UDP(sport=rnd.randint(32768, 60999), dport=53) / ("d" * 40), t + 0.001)
            add(Ether() / IP(src="10.1.1.53", dst=c, ttl=64) / UDP(sport=53, dport=0) / ("a" * 80), t + 0.02)
        t += rnd.expovariate(4.0)

    # SYN scan, minutes 8-9: one source, 6 hosts x 250 ports, ~25 SYN/s, a few RST replies, TTL 47
    att = "203.0.113.66"
    t = 480.0
    for port in range(1, 251):
        for h in range(5, 11):
            sport = rnd.randint(40000, 60000)
            add(Ether() / IP(src=att, dst=f"10.1.1.{h}", ttl=47) / TCP(sport=sport, dport=port, flags="S", seq=1, window=1024), t)
            if rnd.random() < 0.04:
                add(Ether() / IP(src=f"10.1.1.{h}", dst=att, ttl=64) / TCP(sport=port, dport=sport, flags="RA", seq=0, ack=2, window=0), t + 0.002)
            t += 0.04
    # SSH-style burst, minute 10: one source, 300 short connections to 10.1.1.7:22
    t = 600.0
    for _ in range(300):
        session(t, "198.51.100.23", "10.1.1.7", 22, rnd.randint(40000, 60000), nbytes=60, ttl_c=52, ttl_s=64)
        t += rnd.uniform(0.1, 0.3)

    pk.sort(key=lambda p: p.time)
    wrpcap(str(OUT / "demo_traffic.pcap"), pk)
    print(f"demo_traffic.pcap: {len(pk):,} packets, {(max(p.time for p in pk) - T0) / 60:.1f} min")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2 or sys.argv[1] != "pcap-only":
        make_csv(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    make_pcap()

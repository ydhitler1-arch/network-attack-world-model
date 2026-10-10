# SIH PS 26153 - AI based Network Attack Forecasting from Network Traffic Data (NTRO, Software, Blockchain & Cybersecurity)

This repo IS the submission for this problem statement. Requirements below are the checklist to satisfy.

## Core idea
A WORLD MODEL: learn network-state transition dynamics P(S_{t+1} | S_t) from traffic telemetry, roll out K steps ahead,
and estimate the likelihood/progression of infiltration BEFORE the attacker completes the kill chain.
Not a static per-flow benign/malicious classifier.

## Must have
1. Input: open datasets (CIC-IDS2017/2018, CTU-13, UNSW-NB15, CICIoT2023, LANL auth, DARPA...) as CSV flow records and/or PCAP (Scapy/PyShark).
   Features at two levels:
   - flow: src/dst IP+port, TCP flag counts (SYN, ACK, FIN, RST, PSH, URG), protocol, bytes/packets per flow, duration, IAT stats (mean/var/max), bidirectional ratios
   - packet: TTL + variance, TCP window size, IP fragment flags, payload size distribution, port-scan signatures, retransmissions
   Output: timestamped, normalised feature matrix. (Document which packet-level features a CSV dataset cannot provide.)
2. World model: network state as feature vector or graph; learn P(S_{t+1}|S_t) with LSTM / temporal Transformer / GNN over time windows;
   supervised dynamics learning with transitions derived from the dataset's attack timeline; must GENERALISE to unseen attack patterns (not just memorise signatures).
3. Prediction engine: from a current snapshot, K-step forward rollout -> (a) time-series probability of infiltration in next K windows,
   (b) predicted MITRE ATT&CK stage (Reconnaissance, Initial Access, Lateral Movement, Command & Control, Exfiltration),
   (c) top driving features (attention or SHAP).
4. Explainability for EVERY prediction (SHAP or attention). Black-box output is not acceptable.
5. Demo interface (Streamlit/Flask/CLI), fully OFFLINE: accepts a PCAP or CSV, runs inference, shows infiltration-probability timeline, flagged flows, attack-stage annotations.
6. Benchmark vs a LOGISTIC REGRESSION baseline on the same features: F1, precision, recall, false-positive rate; show the temporal world model gives a measurable improvement.
7. Training scripts, model weights, reproducible training config. Fully open source.

## Deliverables for evaluation
Source code link; README with setup instructions; Architecture document (max 2 pages); Demo video (max 2 min); Technical presentation (max 5 slides).

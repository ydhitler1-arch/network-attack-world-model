# Hackathon background (SIH 2026, PS 26153)

This project began as an entry for **SIH 2026 PS 26153: AI-based Network Attack Forecasting from Network Traffic Data**. It was **not submitted**; the work continues as a personal research and portfolio project. The files in this folder are the hackathon-specific material, kept for reference:

- [SIH_PS26153_REQUIREMENTS.md](SIH_PS26153_REQUIREMENTS.md): the requirements checklist the build was measured against
- [PHASE0_AUDIT.md](PHASE0_AUDIT.md): gap analysis and dataset recommendation at the start of the build
- [SLIDES.md](SLIDES.md): 5-slide outline; [SIH_PS26153_deck.pptx](SIH_PS26153_deck.pptx): the built deck
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md): 2-minute demo script

## Status against the problem statement

| # | requirement | status |
|---|---|---|
| 1 | Open datasets as CSV and/or PCAP; flow and packet-level features; timestamped normalised feature matrix | **Done for CIC-IDS2017 CSVs** (43 features per 60 s window, `data/state_v1.parquet`). **PCAP path implemented with Scapy** and tested on a synthetic capture only. A CSV cannot provide TTL, fragmentation flags or retransmissions: the PCAP path extracts them, the trained model does not use them ([../FEATURES.md](../FEATURES.md)). |
| 2 | World model learning P(S_t+1 given S_t) with LSTM, generalising to unseen attacks | **Implemented** (LSTM, Gaussian next-state distribution, trained on dataset timeline transitions). The learned dynamics are **worse than a linear AR(1) predictor**, and generalisation to unseen attack types is **weak** (mean recall 0.19). |
| 3 | K-step rollout: infiltration-probability series, ATT&CK stage, top drivers | **Done** (K = 6). Stage head is weak: **no Exfiltration labels exist** in the data and DoS/DDoS (Impact) fit none of the five stages, so they get a sixth class. |
| 4 | SHAP or attention for every prediction | **Done** (expected-gradients SHAP; validated, ~9% completeness error) ([../EXPLAINABILITY.md](../EXPLAINABILITY.md)). |
| 5 | Offline demo: PCAP or CSV in, timeline, flagged flows, stage annotations | **Done** (`dashboard/demo.py`). |
| 6 | Benchmark vs logistic regression (F1, precision, recall, FPR) showing a measurable improvement | **Benchmark done, improvement not shown**: logistic regression is better. Blocked CV with purge, leave-one-attack-type-out, bootstrap CIs. |
| 7 | Training scripts, weights, reproducible config, open source | **Done**, MIT licensed. |

Deliverables produced: README, 2-page architecture document ([../ARCHITECTURE.pdf](../ARCHITECTURE.pdf)), 5-slide outline and deck, 2-minute demo script, public source repository. Not produced: a recorded demo video.

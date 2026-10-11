# Phase 0 - Audit vs SIH PS 26153 checklist

Facts below come from the repo and from reading the files in `data/raw` (label counts, column lists). Nothing was downloaded.

## 1. Datasets currently available

`data/raw` holds the 8 CIC-IDS2017 "TrafficLabelling" CSVs (85 columns each, with Flow ID, Source/Destination IP and port, Timestamp, per-flow stats, Label). 2,830,743 usable flows after dropping rows with invalid timestamps.

| file | labels (flow counts) |
|---|---|
| Monday | BENIGN 529,918 |
| Tuesday | BENIGN 432,074; FTP-Patator 7,938; SSH-Patator 5,897 |
| Wednesday | BENIGN 440,031; DoS Hulk 231,073; GoldenEye 10,293; slowloris 5,796; Slowhttptest 5,499; Heartbleed 11 |
| Thursday AM | BENIGN 168,186; Web Brute Force 1,507; XSS 652; SQL Injection 21 |
| Thursday PM | BENIGN 288,566; **Infiltration 36** |
| Friday AM | BENIGN 189,067; Bot 1,966 |
| Friday PM DDoS | BENIGN 97,718; DDoS 128,027 |
| Friday PM PortScan | BENIGN 127,537; PortScan 158,930 |

Also present: `data/cleaned_unused/` (Kaggle cleaned file, no timestamps/IPs, unusable), `data/synthetic/` (planted ramp, method demo only).

## 2. Gap analysis

| # | requirement | status | gap |
|---|---|---|---|
| 1 | Flow-level features (IP/port, TCP flag counts, protocol, bytes/packets, duration, IAT stats, bidirectional ratios) | **Partial** | The CSVs contain all of these per flow, but `src/features.py` uses only 13 window aggregates. Per-window flag counts (SYN/ACK/FIN/RST/PSH/URG), IAT mean/std/max, bytes and packets per flow, and fwd/bwd ratios are not in the state vector. |
| 1 | Packet-level features (TTL + variance, TCP window, IP fragment flags, payload size distribution, port-scan signatures, retransmissions) | **Missing** | A CSV cannot provide TTL, fragment flags or retransmissions. It provides only approximations: `Init_Win_bytes_forward/backward` (TCP window of the first packet), packet length min/max/mean/std/variance (a coarse payload-size distribution), `min_seg_size_forward`. Port-scan signatures can be derived from flow aggregates (distinct destination ports per source). Real packet features need a PCAP path. |
| 1 | PCAP input (Scapy/PyShark) | **Missing** | Scapy 2.5.0 is installed but not in `requirements.txt`, and no PCAP code exists. No PCAPs are in the repo (CIC-IDS2017 PCAPs are about 10 GB per day). |
| 1 | Timestamped, normalised feature matrix as documented output | **Partial** | `data/windows.parquet` (2,453 windows, log1p, no per-feature scaling stored). |
| 2 | World model learning P(S_t+1 given S_t) | **Missing** | `src/model.py` is a binary LSTM classifier (`AttackForecastLSTM`) predicting an attack label. There is no next-state prediction, no distributional output, and no transitions derived from the attack timeline. |
| 2 | Generalisation to unseen attacks | **Missing** | Current CV is blocked with purge but there is no leave-one-attack-type-out test. |
| 3 | K-step rollout, infiltration probability series | **Missing** | Only single-horizon classification (and a negative forecasting result). |
| 3 | ATT&CK stage prediction | **Missing** | No stage labels or head. See mapping problems below. |
| 3/4 | Top features / SHAP for every prediction | **Partial** | TreeSHAP for XGBoost only (dashboard shows it for XGBoost alerts). Nothing for the LSTM or world model. |
| 5 | Offline demo accepting CSV/PCAP | **Partial** | `dashboard/app.py` replays precomputed OOF scores only. No upload, no inference on new data, no flagged-flow view, no stage annotations. |
| 6 | Benchmark vs logistic regression | **Partial** | Done at episode level (PR-AUC, detected %, delay) with blocked CV. Missing: F1, precision, recall, false-positive rate; leave-one-attack-type-out; a world model to compare. Honest current result: LSTM (22% episodes detected) does not beat logistic regression (34%). |
| 7 | Training scripts, weights, reproducible config | **Partial** | Training code exists but there is no config file, no fixed seeds, no saved weights checked in. `src/train.py` and `README.md` still describe the synthetic-data forecasting pipeline. |
| - | README with setup | **Stale** | Claims the model catches attacks "during the recon ramp" (true only for the synthetic data). Needs rewriting to say forecasting is a negative result on real data. |
| - | Architecture doc (2 pages), 5-slide deck, 2-minute demo script | **Missing** | `WRITEUP.md` / `WRITEUP.pdf` cover methodology and results for the detection study only. |

Other issues found:
- `requirements.txt` lacks `scapy`; also pin versions for reproducibility.
- The repo is not a git repository.
- `src/__pycache__` has both cp310 and cp313 files (two Python versions have been used here; pin one).

## 3. ATT&CK stage mapping: labels that do not fit

The five required stages are Reconnaissance, Initial Access, Lateral Movement, Command & Control, Exfiltration. Draft mapping for the labels in this data (to be confirmed in Phase 2):

| label | proposed stage | note |
|---|---|---|
| PortScan | Reconnaissance | clean fit |
| FTP-Patator, SSH-Patator | Initial Access | brute force (T1110) |
| Web Brute Force, XSS, SQL Injection | Initial Access | exploit public-facing app (T1190) |
| Heartbleed | Initial Access | only 11 flows, not usable for training or evaluation |
| Bot | Command & Control | botnet beaconing |
| Infiltration | Lateral Movement | the CIC description is a compromised internal host that scans and moves; it may also include Exfiltration, but flows cannot tell the two apart. Only 36 flows. |
| **DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS** | **no stage** | These are Impact (T1498/T1499), not one of the five stages. They are 4 of the 10 attack types and the bulk of attack flows. I propose a sixth "Impact (other)" class kept out of the five-stage output rather than forcing a wrong stage. |
| **Exfiltration** | **no labels** | No CIC-IDS2017 label is exfiltration. The head will have no positive training examples for it; this should be stated, not hidden. |

## 4. Dataset recommendation

**CIC-IDS2017 alone cannot support the world-model claim well:** timestamps are minute-resolution, there are 41 attack episodes, attacks are single-stage and abrupt, and the only multi-stage-like label (Infiltration) has 36 flows.

**Recommendation: add CSE-CIC-IDS2018 as the main dataset for the world model, and keep CIC-IDS2017 as a second dataset for cross-dataset testing.**

Why CIC-IDS2018:
- Timestamps have second resolution, so 5-10 second windows are possible, which is the resolution where a ramp could appear if one exists.
- Many more episodes: about 16M flows over 10 days, with FTP/SSH brute force, DoS (GoldenEye, slowloris, SlowHTTPTest, Hulk), DDoS (LOIC, HOIC), web attacks, Infiltration and Bot.
- The Infiltration scenario (compromise, then internal scanning and further activity) is the closest thing available to a multi-stage chain, with about 160k flows labelled (versus 36 in 2017).
- The feature schema is the same CICFlowMeter family, so most of `src/` is reusable.

Trade-off to know before downloading: as far as I recall, the "Processed Traffic Data for ML Algorithms" CSVs have **no Source IP, Flow ID or Source Port** (only destination port and protocol). That would remove `unique_src_ips`, `top_src_share`, `src_ip_entropy` and `top_pair_share`, which are our strongest 2017 features (about 48% of SHAP importance). Please check the header of the first file you download; if the IP columns are absent, the options are (a) accept the loss and use destination-side and volume features, or (b) use CTU-13, which has IPs.

**Alternative / second choice: CTU-13** (13 botnet scenarios, bidirectional NetFlow with sub-second start times and IPs, labels Background/Normal/Botnet). It has real multi-stage malware behaviour (infection, C&C, scanning, spam) and different malware families per scenario, which suits the unseen-attack test. Limits: labels are botnet vs not, not per stage, and there is no packet-level data in the binetflow files.

### What to download (I have not downloaded anything)

CIC-IDS2018, from the official AWS Open Data bucket (no AWS account needed; check the sizes first, because the full set is large):

```bash
aws s3 ls --no-sign-request --region ap-southeast-2 --human-readable "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
```

Start with these days only (Infiltration and Bot plus brute force give the stage variety; skip the large 02-20 and 02-21 DDoS files at first):

- `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` (Infiltration)
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` (Infiltration)
- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` (Bot)
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` (FTP/SSH brute force)

Put them in a new folder **`data/raw2018/`** (not in `data/raw`, which must stay 2017-only per the gotchas in docs/PROJECT_NOTES.md). File names may differ slightly from what I listed; use whatever the `ls` shows.

If you want real packet-level features (TTL, window, fragmentation, retransmissions) for the demo and the Scapy path, the 2018 raw PCAPs are far too large (hundreds of GB). A small substitute is any short public PCAP, for example a few from the Wireshark sample captures or CTU-13 scenario captures; put them in `data/pcap/`. This is optional and only needed to test the PCAP path.

CTU-13, if you choose it instead: from https://www.stratosphereips.org/datasets-ctu13 take the `.binetflow` file for a few scenarios (suggest 3 different malware families), into `data/ctu13/`.

## 5. Proposed plan changes (for your OK)

1. Keep the honest framing: forecasting is a negative result on CIC-IDS2017, and the world model is evaluated on next-state prediction quality plus early-detection and stage metrics, not on a claim of forecasting.
2. Phase 1 builds the richer state vector from CIC-IDS2017 now; the 2018 data plugs in later through the same loader if you download it.
3. Fix README and `requirements.txt` in Phase 6 (or sooner if you want).

Questions for you before Phase 1: (1) Download CIC-IDS2018 now, or proceed on 2017 only and add 2018 later? (2) Window size: keep 60 s for 2017 (timestamps are minute-resolution anyway).

# Network Attack Early Warning with a Learned World Model

Submission repository for **SIH 2026 PS 26153: AI-based Network Attack Forecasting from Network Traffic Data**.

A world model learns how network state evolves from traffic telemetry, rolls the state forward 6 steps, and reads off a probability of infiltration and an ATT&CK stage at every step, with a SHAP explanation for every prediction and an offline Streamlit demo that accepts a flow CSV or a PCAP.

> **Read this first: what the results say.** Evaluated honestly on CIC-IDS2017, the temporal world model **does not beat a plain logistic-regression baseline** (PR-AUC 0.22 vs 0.45 for detecting the current window), shows **no reliable skill at predicting attacks minutes ahead**, and misses most unseen attack types. The problem statement asks the benchmark to show a measurable improvement from the temporal model; **on this data it does not**. Everything else (pipeline, explanations, demo) works and is validated. Details and caveats: [docs/BENCHMARK.md](docs/BENCHMARK.md), [docs/WORLD_MODEL.md](docs/WORLD_MODEL.md).

## Status against the problem statement

| # | requirement | status |
|---|---|---|
| 1 | Open datasets as CSV and/or PCAP; flow and packet-level features; timestamped normalised feature matrix | **Done for CIC-IDS2017 CSVs** (43 features per 60 s window, `data/state_v1.parquet`). **PCAP path implemented with Scapy** and tested on a synthetic capture only. A CSV cannot provide TTL, fragmentation flags or retransmissions: the PCAP path extracts them, the trained model does not use them ([docs/FEATURES.md](docs/FEATURES.md)). |
| 2 | World model learning P(S_t+1 given S_t) with LSTM, generalising to unseen attacks | **Implemented** (LSTM, Gaussian next-state distribution, trained on dataset timeline transitions). The learned dynamics are **worse than a linear AR(1) predictor**, and generalisation to unseen attack types is **weak** (mean recall 0.19). |
| 3 | K-step rollout: infiltration-probability series, ATT&CK stage, top drivers | **Done** (K = 6). Stage head is weak: **no Exfiltration labels exist** in the data and DoS/DDoS (Impact) fit none of the five stages, so they get a sixth class. |
| 4 | SHAP or attention for every prediction | **Done** (expected-gradients SHAP; validated, ~9% completeness error) ([docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md)). |
| 5 | Offline demo: PCAP or CSV in, timeline, flagged flows, stage annotations | **Done** (`dashboard/demo.py`). |
| 6 | Benchmark vs logistic regression (F1, precision, recall, FPR) showing a measurable improvement | **Benchmark done, improvement not shown**: logistic regression is better. Blocked CV with purge, leave-one-attack-type-out, bootstrap CIs. |
| 7 | Training scripts, weights, reproducible config, open source | **Done**, MIT licensed ([LICENSE](LICENSE)). |

Deliverables: README (this file), architecture document ([docs/ARCHITECTURE.pdf](docs/ARCHITECTURE.pdf), 2 pages), 5-slide outline ([docs/SLIDES.md](docs/SLIDES.md)), 2-minute demo script ([docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)). **Not produced:** the recorded video, the built slide deck, and the public source link (the local git repository exists but has no commits and no remote yet).

## Setup

Tested with Python 3.13.5 on Windows 11, CPU only.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Quick start: the demo only (no dataset needed)

The repository ships the trained weights and two samples (`data/sample/`: a 55-minute slice of real CIC-IDS2017 traffic and a synthetic PCAP).

```bash
streamlit run dashboard/demo.py
```

Pick a sample in the sidebar, or upload your own CSV (CICFlowMeter layout) or PCAP. Notes: the CIC-IDS2017 sample is **in-sample** for the shipped demo model; the PCAP sample is **synthetic**, and the model does not flag the scan in it.

### Full reproduction

1. Download CIC-IDS2017 from the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/ids-2017.html) and put the eight flow CSVs that include `Flow ID`, IP addresses and timestamps (file names end in `.pcap_ISCX.csv`) in `data/raw/`. Do not use the "cleaned" Kaggle version (no timestamps or IPs).
2. Run, in order:

```bash
python -W ignore src/state.py data/raw 60        # flows -> 43-feature windows (a few minutes, several GB RAM)
python -W ignore src/wm_train.py                 # train/validate world model, sanity evaluation (~1 min)
python -W ignore src/wm_train.py --mode full     # demo weights trained on all days
python -W ignore src/benchmark.py --part cv --seeds 3     # blocked CV vs baselines (~1.5 min)
python -W ignore src/benchmark.py --part loto             # leave-one-attack-type-out (~6 min)
python -W ignore src/benchmark.py --part report           # metrics + bootstrap CIs -> models/benchmark_results.json
python -W ignore src/explain.py                  # SHAP table and validation (~4 min)
python -W ignore src/make_demo_data.py data/raw  # rebuild data/sample/ (optional)
```

3. Tests: `python -W ignore tests/test_phase1.py` (PCAP to state vector) and `python -W ignore tests/test_phase5.py` (demo pipeline on the samples).
4. Rebuild the architecture PDF: `python docs/make_architecture_figure.py` then `python docs/md2pdf.py docs/ARCHITECTURE.md docs/ARCHITECTURE.pdf` (needs matplotlib and reportlab).

Seeds are fixed (42) and training is deterministic on CPU: repeated runs gave identical metrics. Hyperparameters and the ATT&CK stage mapping are in [configs/world_model.json](configs/world_model.json).

## Repository layout

| path | contents |
|---|---|
| `src/data_loader.py` | CIC-IDS2017 CSV loader (per-file timestamp handling, optional extended columns) |
| `src/state.py` | 43-feature state vector, scaler |
| `src/pcap_to_flows.py` | PCAP to flows with Scapy, adds TTL / fragmentation / retransmission columns |
| `src/world_model.py`, `src/wm_train.py` | world model, rollout, training and sanity evaluation |
| `src/benchmark.py` | blocked CV with purge, leave-one-attack-type-out, bootstrap |
| `src/explain.py`, `src/inference.py` | SHAP, counterfactual host analysis, demo pipeline |
| `dashboard/demo.py` | offline Streamlit demo; `dashboard/app.py` is the older replay dashboard for the early-detection study |
| `configs/world_model.json` | seeds, hyperparameters, stage mapping |
| `models/` | weights (`world_model.pt`, `world_model_full.pt`), metrics, explanations |
| `docs/` | architecture, features, world model, benchmark, explainability, slides, demo script |
| `data/sample/` | demo inputs; `data/state_v1.parquet` is the derived feature matrix |
| `WRITEUP.md` / `.pdf` | the earlier study: forecasting is at chance, reframed as early detection |
| `src/features.py`, `evaluate.py`, `detect_eval.py`, `train.py`, `model.py`, `sequence_builder.py`, `make_synthetic_data.py` | the original pipeline and the synthetic-data demo (a planted recon ramp, trivially separable, a methodology demo only) |

## Results summary

Detecting the current window, blocked 5-fold CV, thresholds calibrated for a 2% false-alarm rate on a validation fold (2,095 test samples, 11% attack):

| model | PR-AUC [95% CI] | precision | recall | F1 | FPR |
|---|---|---|---|---|---|
| Logistic regression | **0.451** [0.23, 0.67] | 0.67 | 0.32 | **0.43** | 0.020 |
| LSTM classifier | 0.350 [0.16, 0.59] | 0.32 | 0.30 | 0.31 | 0.078 |
| World model | 0.219 [0.11, 0.37] | 0.25 | 0.17 | 0.20 | 0.061 |

- Forecasting 3 minutes ahead: all models PR-AUC 0.18-0.25 against a base rate of 0.11, no significant differences.
- Leave-one-attack-type-out, mean recall: logistic regression 0.36, LSTM 0.25, world model 0.19; Bot, DoS GoldenEye and DoS Hulk are missed by every model.
- Next-state error: the world model beats persistence but loses to a linear AR(1) predictor at every horizon.
- Explanations pass the faithfulness check (deleting top-SHAP features lowers the score, random features do not).

## Known limitations

Only 41 attack episodes in 5 days of one lab network; minute-resolution timestamps; the demo model trained on all days for 3 epochs; scripted attacks with no realistic reconnaissance phase; no Exfiltration or multi-stage kill-chain labels; thresholds drift across days (realised false-alarm rates 2-8% against a 2% target). The suggested next step is a finer-resolution dataset with multi-stage attacks (CIC-IDS2018, CTU-13) and stronger regularisation of the world model; neither has been tried.

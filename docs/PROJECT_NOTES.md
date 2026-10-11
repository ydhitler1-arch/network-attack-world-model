# Network Attack Early Detection (CIC-IDS2017) - project context

This repo began as an SIH 2026 entry for PS 26153 (AI-based network attack forecasting, NTRO) but was NOT submitted; it is now a personal research/portfolio project.
Hackathon material (requirements checklist, audit, slides, demo script) is in docs/hackathon/. All checklist items were implemented; benchmark shows the world model does NOT beat the LR baseline (see below).
Owner: Devan (cybersecurity student). Python 3.13 on Windows, torch already installed.

## Pipeline
flows -> 60s windows (13 features, log1p) -> last-N-minute sequences -> models -> blocked CV with purge -> detection delay at fixed false-alarm rate.
- src/data_loader.py  loads CIC-IDS2017 "GeneratedLabelledFlows" CSVs (cp1252, per-file day-first timestamps, hours 1-7 = PM, blank rows dropped)
- src/features.py     60s windows + segments at overnight gaps; usage: python src/features.py data/raw 60
- src/legacy/evaluate.py     forecasting experiment (+ --detect, --by-type, --cv-episodes)
- src/legacy/detect_eval.py  early-detection eval: anomaly heuristic, logistic regression, XGBoost (TreeSHAP), optional LSTM (--lstm)
- dashboard/app.py    Streamlit; reads models/detection_oof.parquet (streamlit run dashboard/app.py)

## Findings so far (data/windows_v1_8feat.parquet = old 8-feature windows)
- Forecasting (attack within next 6 min) is at chance on CIC-IDS2017 at 1-min resolution: attacks start abruptly, no ramp in the 6 windows before onset (DDoS, Hulk, PortScan checked). Only 41 episodes; timestamps are minute-resolution.
- Detection with the 8 original features: ~20% of episodes caught at 2% false-alarm rate; simple anomaly heuristic ~ learned models. Features see only DDoS/Hulk/PortScan; Bot, Patator, web attacks (7-10% of window flows) are invisible.
- Decision: reframe as EARLY DETECTION (delay after onset), keep the forecasting negative result in the write-up.

## Submission build (Phases 0-6 done; see README.md for the status table and commands)
- src/state.py (43-feature state), src/pcap_to_flows.py (Scapy), src/world_model.py + wm_train.py (LSTM world model, K=6 rollout, configs/world_model.json),
  src/benchmark.py (blocked CV + leave-one-type-out), src/explain.py + inference.py (SHAP, host counterfactuals), dashboard/demo.py (offline demo).
- Docs: docs/ARCHITECTURE.pdf (2 pages), BENCHMARK.md, WORLD_MODEL.md, EXPLAINABILITY.md, FEATURES.md, BLOG.md (narrative write-up), EARLY_DETECTION_STUDY.md; docs/hackathon/ has the SIH slides and demo script.
- Benchmark result (honest, revised objective): world model PR-AUC 0.30 < LSTM classifier 0.35 < logistic regression 0.45 for detection (differences not significant); no established forecasting skill; weak on unseen attack types. Original objective scored 0.22 (significantly worse); see docs/OVERFITTING_TEST.md and models/v1_original_objective/.
  The PS asks to show the temporal model improves on LR: NOT met on CIC-IDS2017.
- Earlier steps (features.py -> legacy/detect_eval.py --lstm -> dashboard/app.py -> docs/EARLY_DETECTION_STUDY.md) are finished.

## Still open
- Portfolio polish done (README, screenshots, docs/BLOG.md). Repo is public: https://github.com/ydhitler1-arch/network-attack-world-model (MIT).
- Optional: CIC-IDS2018 / CTU-13 (finer time resolution, multi-stage), stronger regularisation of the world model.

## Gotchas
- Run steps as separate commands; if features.py fails, later steps silently use stale data/windows.parquet.
- data/raw must contain only the timestamped GeneratedLabelledFlows CSVs (NOT the Kaggle "cleaned" file: no timestamps/IPs).
- Synthetic data (data/synthetic, src/legacy/make_synthetic_data.py) has a planted recon ramp and is trivially separable: methodology demo only.
- Don't claim forecasting works; results on real data say otherwise.

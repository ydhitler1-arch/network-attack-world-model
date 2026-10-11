# Changelog

## v1.0.0 (2026-10-12)

First complete release: a reproducible, explainable network-intrusion pipeline and an honest evaluation of an LSTM
"world model" on CIC-IDS2017.

### What is in it
- **Pipeline:** 43-feature 60 s state vector from flow CSVs; PCAP-to-flows path with Scapy (adds TTL, fragmentation and
  retransmission columns the CSVs cannot provide).
- **World model:** LSTM with a 6-step rollout, infiltration and ATT&CK-stage heads, trained with the infiltration and
  stage losses on real history steps only (`rollout_task_loss: false`).
- **Explanations:** SHAP (expected-gradients) for every prediction, validated against the `shap` library, with a deletion
  test and a measured completeness error; counterfactual host analysis.
- **Offline demo:** Streamlit app for CSV or PCAP input; Docker image.
- **Evaluation:** blocked cross-validation with a purge gap, leave-one-attack-type-out, block-bootstrap intervals,
  paired comparisons, and tests that the splits do not leak.
- **Follow-up studies:** why the world model loses ([docs/OVERFITTING_TEST.md](docs/OVERFITTING_TEST.md)), threshold
  drift and its causes ([docs/THRESHOLD_DRIFT.md](docs/THRESHOLD_DRIFT.md), [docs/DRIFT_CAUSES.md](docs/DRIFT_CAUSES.md),
  [docs/VOLUME_OOD.md](docs/VOLUME_OOD.md)), attempts to improve the baseline
  ([docs/IMPROVING_THE_BASELINE.md](docs/IMPROVING_THE_BASELINE.md)).
- **CI:** lint (ruff), tests, and a Docker build with a serve check on every push.

### Headline results
- Detecting the current window, PR-AUC: logistic regression 0.451, LSTM classifier 0.350, world model 0.299 (the
  differences are not statistically significant with 41 attack episodes; the world model has the lowest point estimate).
- No established forecasting skill 3 minutes ahead; weak recall on unseen attack types (mean 0.21 for the world model).
- Neural models' alert thresholds drift between days (6.5% to 7.8% false alarms against a 2% target); logistic regression
  does not (2.0%). Delayed analyst feedback roughly halves the excess.
- Nothing tried improved on plain logistic regression under a clean selection protocol.

### Known limitations
41 attack episodes in 5 days of one lab network; minute-resolution timestamps; the demo model was trained on all days
for 4 epochs, so the bundled CIC-IDS2017 sample is in-sample; the stage head cannot output Exfiltration (no labels);
the PCAP path has been tested on synthetic traffic only.

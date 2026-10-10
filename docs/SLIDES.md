# 5-slide outline: World-model early warning of network infiltration (SIH PS 26153)

Framing for the whole deck: we built the complete system the problem statement describes, evaluated it honestly, and the headline finding is a **measured negative result**: on CIC-IDS2017 the temporal world model does not beat logistic regression. Judges will respect numbers that are real more than a claim that cannot be checked.

## Slide 1: Problem and approach

**Title:** From static classification to a learned model of network state

- Reactive IDS classifies flows after the fact. PS 26153 asks for a model of how network state evolves, so infiltration risk can be estimated before the kill chain completes.
- Our system: traffic to 60 s state vector, LSTM world model learns P(next state), 6-step rollout gives P(infiltration) and ATT&CK stage, SHAP explains every output.
- Built, tested and benchmarked end to end; fully offline.

**Visual:** `docs/architecture.png`.
**Notes:** Say up front that the evaluation is honest, including where the idea does not work.

## Slide 2: Data and state vector

**Title:** CIC-IDS2017 to a 43-feature state per minute

- 2.83M flows, 2,453 windows of 60 s, 264 attack windows, 41 attack episodes over 5 days.
- 43 features: flow statistics, TCP flag shares, IAT, backward/forward ratios, scan signatures (ports and hosts per source), concentration shares, TCP-window and packet-size proxies.
- PCAP path (Scapy) adds TTL and variance, fragmentation, retransmissions. A CSV cannot provide these, so the trained model does not use them.
- Normalisation fit on benign training windows only; attacks start abruptly at minute resolution.

**Visual:** feature groups table from `docs/FEATURES.md`.
**Notes:** Mention the earlier finding: no precursor signal before attacks (DDoS, Hulk, PortScan checked), so "forecasting" at this resolution is at chance.

## Slide 3: World model, prediction engine, explainability

**Title:** Rollout, ATT&CK stage, and SHAP on every prediction

- LSTM (100,702 parameters) outputs a Gaussian over the next state plus infiltration and stage heads; reads 6 windows, rolls out 6 steps.
- Stage mapping: PortScan = Recon, Patator/web attacks = Initial Access, Bot = C2, Infiltration = Lateral Movement. DoS/DDoS are Impact (outside the five stages). No label is Exfiltration, so that head cannot learn it.
- SHAP (expected gradients through the rollout): r = 0.99 with the `shap` library; deleting the top 3 features drops the logit 1.27 vs about 0 for random features; 8.8% completeness error.
- Host-level counterfactual: removing the DDoS attacker or victim from the lookback drops P from 0.98 to about 0.04.

**Visual:** demo screenshot of the SHAP bar chart and host table.
**Notes:** Be explicit about what is approximate (completeness) and what is missing (Exfiltration).

## Slide 4: Results: an honest benchmark

**Title:** The temporal model does not beat logistic regression here

| detect current window | PR-AUC [95% CI] | F1 |
|---|---|---|
| Logistic regression | 0.451 [0.23, 0.67] | 0.43 |
| LSTM classifier | 0.350 [0.16, 0.59] | 0.31 |
| World model | 0.219 [0.11, 0.37] | 0.20 |

- Blocked 5-fold CV with purge, scaler refit per fold, thresholds from a validation fold, 3 seeds, block-bootstrap CIs.
- Forecasting 3 minutes ahead: PR-AUC 0.18-0.25 for all models (base rate 0.11), no significant differences.
- Unseen attack types (leave-one-out): mean recall 0.36 logistic regression vs 0.19 world model.
- Learned dynamics lose to a linear AR(1) predictor.

**Visual:** bar chart of PR-AUC with CIs (k = 0 and k = 3).
**Notes:** Likely cause (hypothesis, untested): a 100k-parameter model overfits about 1,500 overlapping samples; 41 episodes is too little for dynamics learning.

## Slide 5: Demo, limits, next steps

**Title:** What works, what does not, what next

- Live demo: upload a CSV or PCAP, see the probability timeline, ATT&CK stage markers, flagged windows with drivers, flagged hosts and flows. No network needed.
- Limits: 41 episodes, one lab network, minute resolution, demo weights trained on all days (sample is in-sample), thresholds drift across days.
- Next: CIC-IDS2018 or CTU-13 (finer time resolution, multi-stage attacks), stronger regularisation, host-level graph state, per-attack-type thresholds.
- Reproducible: fixed seeds, config, weights, tests, pinned requirements.

**Visual:** demo timeline screenshot.
**Notes:** Close with the contribution: a reproducible, explainable baseline and an evaluation protocol (purged blocked CV, leave-one-attack-out) that exposes weaknesses instead of hiding them.

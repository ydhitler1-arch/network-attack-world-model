# Benchmark (Phase 3)

Code: `src/benchmark.py`. Raw predictions: `models/bench_cv.pkl`, `models/bench_loto.pkl`. All numbers below: `models/benchmark_results.json`.

## Setup

- Same 43-feature state (Phase 1), same samples (last 6 windows) for all models: logistic regression on the current window (`LogReg current`), logistic regression on a 6-window summary (last / mean / max / slope, 172 features; `LogReg 6-win`), an LSTM classifier (same architecture and hyperparameters as the world model, dynamics head removed), and the world model.
- Blocked 5-fold CV: blocks of 60 windows, block b in fold b mod 5; one other fold is the inner validation set (early stopping and threshold calibration); samples never straddle regions or capture days; windows within 12 of a test/validation window are purged from training; scaler refit per fold on benign training windows. 2,095 test samples, 11.0% attack windows.
- **Operating point:** the threshold is calibrated on the validation fold's benign windows for a 2% false-alarm rate, then applied to the test fold. So precision / recall / F1 / FPR are what you would get without peeking at test labels. `recall at 2% FPR (oracle)` re-calibrates on the pooled test scores and is optimistic.
- Neural models: 3 seeds, mean reported. 95% CIs come from a block bootstrap (resampling 60-window blocks).
- Task k=0 detects the current window. Task k=3 predicts the label 3 minutes ahead (world model: 3-step rollout; others trained on the shifted label).
- Leave-one-attack-type-out (LOTO): every window of the held-out type (plus a 12-window margin) is removed from training and validation; positives are the held-out type's windows, negatives are benign windows; other attack types are excluded from that metric. One seed per fold, 5 folds.

## Results: blocked CV

**k=0, detect the current window** (attack rate 11.0%)

| model | PR-AUC [95% CI] | ROC-AUC | precision | recall | F1 [95% CI] | FPR | recall at 2% FPR (oracle) |
|---|---|---|---|---|---|---|---|
| LogReg current | **0.451** [0.23, 0.67] | **0.725** | 0.667 | 0.322 | **0.434** [0.23, 0.62] | 0.020 | 0.296 |
| LogReg 6-win | 0.440 [0.21, 0.66] | 0.695 | 0.506 | 0.348 | 0.412 [0.24, 0.58] | 0.042 | 0.291 |
| LSTM classifier | 0.350 [0.16, 0.59] | 0.710 | 0.321 | 0.300 | 0.310 [0.10, 0.52] | 0.078 | 0.230 |
| World model | 0.219 [0.11, 0.37] | 0.629 | 0.251 | 0.167 | 0.200 [0.05, 0.36] | 0.061 | 0.104 |

**k=3, predict 3 minutes ahead**

| model | PR-AUC [95% CI] | ROC-AUC | precision | recall | F1 [95% CI] | FPR | recall at 2% FPR (oracle) |
|---|---|---|---|---|---|---|---|
| LogReg current | 0.250 [0.09, 0.41] | 0.599 | 0.477 | 0.221 | 0.302 [0.12, 0.46] | 0.030 | 0.126 |
| LogReg 6-win | 0.180 [0.08, 0.32] | 0.537 | 0.333 | 0.139 | 0.196 [0.07, 0.32] | 0.034 | 0.087 |
| LSTM classifier | 0.215 [0.08, 0.34] | 0.621 | 0.217 | 0.139 | 0.169 [0.03, 0.29] | 0.062 | 0.095 |
| World model | 0.217 [0.08, 0.36] | 0.611 | 0.286 | 0.162 | 0.206 [0.04, 0.35] | 0.051 | 0.133 |

**Paired difference, world model minus baseline** (block bootstrap, 95% CI)

| baseline | k=0 PR-AUC | k=0 F1 | k=3 PR-AUC | k=3 F1 |
|---|---|---|---|---|
| LogReg current | -0.222 [-0.41, -0.05] | -0.232 [-0.45, -0.04] | -0.030 [-0.12, +0.05] | -0.089 [-0.22, +0.02] |
| LogReg 6-win | -0.211 [-0.36, -0.07] | -0.212 [-0.38, -0.06] | +0.029 [-0.07, +0.12] | +0.002 [-0.10, +0.09] |
| LSTM classifier | -0.122 [-0.28, -0.01] | -0.105 [-0.21, -0.01] | +0.006 [-0.03, +0.05] | +0.035 [-0.00, +0.08] |

## Results: leave-one-attack-type-out (k=0)

Cells: PR-AUC / recall / FPR at the validation-calibrated threshold.

| held-out type | pos | LogReg current | LogReg 6-win | LSTM classifier | World model |
|---|---|---|---|---|---|
| Bot | 34 | 0.01 / 0.00 / 0.02 | 0.02 / 0.00 / 0.04 | 0.01 / 0.00 / 0.06 | 0.01 / 0.00 / 0.06 |
| DDoS | 16 | 0.35 / 0.94 / 0.03 | 0.95 / 1.00 / 0.04 | 0.84 / 1.00 / 0.06 | 0.87 / 0.94 / 0.05 |
| DoS GoldenEye | 8 | 0.00 / 0.00 / 0.02 | 0.00 / 0.00 / 0.04 | 0.00 / 0.00 / 0.06 | 0.01 / 0.00 / 0.05 |
| DoS Hulk | 20 | 0.01 / 0.05 / 0.02 | 0.01 / 0.00 / 0.04 | 0.02 / 0.00 / 0.07 | 0.01 / 0.00 / 0.05 |
| DoS Slowhttptest | 9 | 0.35 / 0.78 / 0.02 | 0.00 / 0.00 / 0.04 | 0.01 / 0.00 / 0.08 | 0.01 / 0.00 / 0.05 |
| DoS slowloris | 23 | 0.36 / 0.78 / 0.02 | 0.33 / 0.87 / 0.04 | 0.02 / 0.00 / 0.07 | 0.02 / 0.00 / 0.04 |
| FTP-Patator | 52 | 0.21 / 0.25 / 0.03 | 0.19 / 0.37 / 0.04 | 0.35 / 0.35 / 0.04 | 0.04 / 0.08 / 0.05 |
| PortScan | 14 | 0.29 / 0.57 / 0.03 | 0.02 / 0.43 / 0.04 | 0.01 / 0.00 / 0.05 | 0.01 / 0.00 / 0.04 |
| SSH-Patator | 51 | 0.05 / 0.10 / 0.03 | 0.04 / 0.14 / 0.04 | 0.02 / 0.00 / 0.06 | 0.02 / 0.00 / 0.05 |
| Web Brute Force | 12 | 0.05 / 0.08 / 0.02 | 0.06 / 0.42 / 0.04 | 0.11 / 0.42 / 0.06 | 0.05 / 0.25 / 0.05 |
| Web XSS | 5 | 0.06 / 0.40 / 0.02 | 0.06 / 0.80 / 0.04 | 0.07 / 1.00 / 0.07 | 0.11 / 0.80 / 0.05 |
| **mean over types** | | 0.159 / 0.359 / 0.024 | 0.153 / 0.365 / 0.040 | 0.134 / 0.251 / 0.062 | 0.106 / 0.188 / 0.049 |

## Does the temporal world model help? No, on this data.

1. **Detection (k=0): it is the worst model.** The world model's PR-AUC is 0.22 against 0.45 for logistic regression on the single current window. The paired CIs exclude zero against all three baselines. The simple non-temporal model wins; adding history (`LogReg 6-win`, LSTM) did not help either.
2. **Forecasting 3 minutes ahead: nothing measurable.** All models are only modestly above the 11% base rate (PR-AUC 0.18-0.25, ROC-AUC 0.54-0.62). Differences between the world model and the baselines are within noise (every CI includes 0, except that LSTM vs world model on F1 touches 0 at -0.00). This repeats the earlier negative forecasting result with the new features and models.
3. **Unseen attack types: weak for everyone, worst for the world model.** Mean recall over the 11 held-out types is 0.36 for logistic regression, 0.25 for the LSTM, and 0.19 for the world model. Bot, DoS GoldenEye and DoS Hulk are missed by all models (recall about 0). Only DDoS is caught well, and by the history-based models (recall 0.94-1.00). Per-type results are noisy (5-52 positive windows, one seed).
4. **False alarms drift above target.** Calibrated for 2% FPR on validation, the realised FPR on test is 2.0% for `LogReg current`, 4.2% for `LogReg 6-win`, 7.8% for the LSTM and 6.1% for the world model: scores shift between days, so a fixed threshold does not hold.

## Caveats

- 41 episodes and 5 capture days: wide CIs; the ranking of the two logistic regressions and the LSTM is not reliable.
- The world model was not tuned beyond the Phase 2 default configuration, and neither were the baselines; a better-regularised or differently weighted model (for example a lower dynamics-loss weight) might close part of the gap. I did not test that, so the cause of the gap is a hypothesis: the likely candidate is overfitting of a 100k-parameter model on about 1,500 overlapping samples.
- Early stopping used the validation infiltration loss for both neural models.
- Stage prediction is not benchmarked (the label set cannot support it, see `docs/WORLD_MODEL.md`).
- Thresholds are per fold; the realised FPRs above show they are not stable across days.

## Reproduce

```bash
python -W ignore src/benchmark.py --part cv --seeds 3     # about 1.5 min
python -W ignore src/benchmark.py --part loto             # about 6 min
python -W ignore src/benchmark.py --part report
```

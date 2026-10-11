# Benchmark (Phase 3)

Code: `src/benchmark.py`. Raw predictions: `models/bench_cv.pkl`, `models/bench_loto.pkl` (not tracked in git). All numbers below: `models/benchmark_results.json`.

> **Version note.** These are the results for the **revised world-model objective** (the infiltration and stage losses applied to real history steps only; see [WORLD_MODEL.md](WORLD_MODEL.md) and [OVERFITTING_TEST.md](OVERFITTING_TEST.md)). The baselines are unchanged and bit-identical to the first run. The first run, with the original objective, is kept in `models/v1_original_objective/benchmark_results.json`: there the world model scored PR-AUC 0.219 at k=0, significantly below logistic regression (paired difference -0.22, CI [-0.41, -0.05]).

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
| World model | 0.299 [0.14, 0.53] | 0.710 | 0.335 | 0.265 | 0.295 [0.09, 0.50] | 0.065 | 0.174 |

**k=3, predict 3 minutes ahead**

| model | PR-AUC [95% CI] | ROC-AUC | precision | recall | F1 [95% CI] | FPR | recall at 2% FPR (oracle) |
|---|---|---|---|---|---|---|---|
| LogReg current | 0.250 [0.09, 0.41] | 0.599 | 0.477 | 0.221 | 0.302 [0.12, 0.46] | 0.030 | 0.126 |
| LogReg 6-win | 0.180 [0.08, 0.32] | 0.537 | 0.333 | 0.139 | 0.196 [0.07, 0.32] | 0.034 | 0.087 |
| LSTM classifier | 0.215 [0.08, 0.34] | 0.621 | 0.217 | 0.139 | 0.169 [0.03, 0.29] | 0.062 | 0.095 |
| World model | 0.267 [0.10, 0.44] | 0.680 | 0.333 | 0.206 | 0.254 [0.07, 0.41] | 0.052 | 0.147 |

**Paired difference, world model minus baseline** (block bootstrap, 95% CI)

| baseline | k=0 PR-AUC | k=0 F1 | k=3 PR-AUC | k=3 F1 |
|---|---|---|---|---|
| LogReg current | -0.143 [-0.35, +0.05] | -0.141 [-0.39, +0.07] | +0.018 [-0.09, +0.11] | -0.044 [-0.19, +0.06] |
| LogReg 6-win | -0.132 [-0.29, +0.00] | -0.121 [-0.36, +0.07] | +0.076 [-0.02, +0.19] | +0.048 [-0.07, +0.16] |
| LSTM classifier | -0.042 [-0.10, +0.01] | -0.014 [-0.04, +0.01] | +0.054 [-0.01, +0.14] | **+0.080 [+0.01, +0.16]** |

Twelve paired comparisons are shown; one interval excludes zero (k=3 F1 against the LSTM classifier). With that many comparisons, one such result is not strong evidence on its own.

## Results: leave-one-attack-type-out (k=0)

Cells: PR-AUC / recall / FPR at the validation-calibrated threshold.

| held-out type | pos | LogReg current | LogReg 6-win | LSTM classifier | World model |
|---|---|---|---|---|---|
| Bot | 34 | 0.01 / 0.00 / 0.02 | 0.02 / 0.00 / 0.04 | 0.01 / 0.00 / 0.06 | 0.01 / 0.00 / 0.06 |
| DDoS | 16 | 0.35 / 0.94 / 0.03 | 0.95 / 1.00 / 0.04 | 0.84 / 1.00 / 0.06 | 0.69 / 0.94 / 0.06 |
| DoS GoldenEye | 8 | 0.00 / 0.00 / 0.02 | 0.00 / 0.00 / 0.04 | 0.00 / 0.00 / 0.06 | 0.00 / 0.00 / 0.06 |
| DoS Hulk | 20 | 0.01 / 0.05 / 0.02 | 0.01 / 0.00 / 0.04 | 0.02 / 0.00 / 0.07 | 0.01 / 0.00 / 0.05 |
| DoS Slowhttptest | 9 | 0.35 / 0.78 / 0.02 | 0.00 / 0.00 / 0.04 | 0.01 / 0.00 / 0.08 | 0.01 / 0.00 / 0.06 |
| DoS slowloris | 23 | 0.36 / 0.78 / 0.02 | 0.33 / 0.87 / 0.04 | 0.02 / 0.00 / 0.07 | 0.02 / 0.00 / 0.08 |
| FTP-Patator | 52 | 0.21 / 0.25 / 0.03 | 0.19 / 0.37 / 0.04 | 0.35 / 0.35 / 0.04 | 0.28 / 0.33 / 0.04 |
| PortScan | 14 | 0.29 / 0.57 / 0.03 | 0.02 / 0.43 / 0.04 | 0.01 / 0.00 / 0.05 | 0.01 / 0.00 / 0.05 |
| SSH-Patator | 51 | 0.05 / 0.10 / 0.03 | 0.04 / 0.14 / 0.04 | 0.02 / 0.00 / 0.06 | 0.03 / 0.00 / 0.04 |
| Web Brute Force | 12 | 0.05 / 0.08 / 0.02 | 0.06 / 0.42 / 0.04 | 0.11 / 0.42 / 0.06 | 0.08 / 0.25 / 0.06 |
| Web XSS | 5 | 0.06 / 0.40 / 0.02 | 0.06 / 0.80 / 0.04 | 0.07 / 1.00 / 0.07 | 0.09 / 0.80 / 0.06 |
| **mean over types** | | 0.159 / 0.359 / 0.024 | 0.153 / 0.365 / 0.040 | 0.134 / 0.251 / 0.062 | 0.113 / 0.210 / 0.055 |

## Does the temporal world model help? Not shown on this data.

1. **Detection (k=0): the world model is no longer significantly worse, but it is not better.** PR-AUC 0.30 against 0.45 for logistic regression on the single current window and 0.35 for the LSTM classifier. All paired intervals now include zero (the closest, against the 6-window logistic regression, is -0.132 [-0.29, +0.00]), so with 41 episodes the three cannot be separated, but the world model has the lowest point estimate of the four. Adding history (`LogReg 6-win`, LSTM, world model) never beat the single-window logistic regression.
2. **Forecasting 3 minutes ahead: the world model has the highest point estimate, but nothing is established.** PR-AUC 0.27 against 0.18-0.25 for the others (base rate 0.11), ROC-AUC 0.68 against 0.54-0.62. Only 1 of the 12 paired intervals excludes zero (F1 against the LSTM classifier, +0.080 [+0.01, +0.16]); the PR-AUC differences all include zero. Earlier experiments found no precursor signal at one-minute resolution, so treat this as a hint to follow up with finer-resolution data, not as evidence that the world model forecasts.
3. **Unseen attack types: weak for everyone.** Mean recall over the 11 held-out types is 0.36 for logistic regression, 0.25 for the LSTM, and 0.21 for the world model. Bot, DoS GoldenEye and DoS Hulk are missed by all models (recall about 0). DDoS is caught well by the history-based models (recall 0.94-1.00). The world model gained on FTP-Patator (PR-AUC 0.28, was 0.04) and lost on DDoS (0.69, was 0.87). Per-type results are noisy (5-52 positive windows, one seed).
4. **False alarms drift above target.** Calibrated for 2% FPR on validation, the realised FPR on test is 2.0% for `LogReg current`, 4.2% for `LogReg 6-win`, 7.8% for the LSTM and 6.5% for the world model: scores shift between days, so a fixed threshold does not hold.

The requirement that the temporal model show a measurable improvement over logistic regression is **not met**: at detection the world model is behind (not significantly) and at 3-minute forecasting it is ahead by an amount that is not distinguishable from noise.

## Caveats

- 41 episodes and 5 capture days: wide CIs; the ordering of the logistic regressions, the LSTM and the world model is not reliable.
- **How the objective was chosen.** The first run used the original world-model objective (also training the infiltration head on imagined future states) and found the model significantly worse than logistic regression. [OVERFITTING_TEST.md](OVERFITTING_TEST.md) shows that model size and regularisation did not explain it, and that the training objective did. The revised objective was selected by validation PR-AUC, not test, and then the full benchmark was re-run; the baselines were not changed. The original results are kept in `models/v1_original_objective/`. This is still a decision made after seeing a poor result, so the numbers above are best read as "what the world model achieves with a sensible objective", not as an untouched pre-registered test.
- Early stopping used the validation infiltration loss for both neural models.
- Stage prediction is not benchmarked (the label set cannot support it, see [WORLD_MODEL.md](WORLD_MODEL.md)).
- Thresholds are per fold; the realised FPRs above show they are not stable across days.

## Reproduce

```bash
python -W ignore src/benchmark.py --part cv --seeds 3     # about 1.5 min
python -W ignore src/benchmark.py --part loto             # about 6 min
python -W ignore src/benchmark.py --part report
```

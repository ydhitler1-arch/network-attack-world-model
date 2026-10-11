# World model (Phase 2)

Code: `src/world_model.py` (model, rollout, inference), `src/wm_train.py` (training, sanity evaluation), config `configs/world_model.json` (seed 42, CPU, all hyperparameters, stage mapping). Weights: `models/world_model.pt` (trained on Mon-Wed, early-stopped on Thu), `models/world_model_full.pt` (all days, for the demo only).

> **Version note.** The shipped weights and every number in this document use the **revised training objective** (`rollout_task_loss: false`, see below). The original objective, which also trained the infiltration and stage heads on imagined future states, scored worse; its results are kept in `models/v1_original_objective/` and explained in [OVERFITTING_TEST.md](OVERFITTING_TEST.md).

## Design

- Input: normalised 43-feature state S_t per 60 s window (Phase 1). Lookback L = 6 windows, rollout horizon K = 6.
- One LSTM cell (128 hidden, 100,702 parameters in total). After consuming a state it outputs:
  1. **dynamics**: a diagonal Gaussian over the next state, `N(x + delta(h), sigma(h)^2)`
  2. **infiltration probability** for the window it just consumed
  3. **ATT&CK stage** (7 classes) for that window
- **K-step rollout:** after the 6 real windows, the predicted mean state is fed back as the next input, 6 times. The infiltration and stage heads are read at every step, giving a probability series for the current window (k=0) and the next 6 (k=1..6), each read from an imagined state. The dynamics loss is trained through the same rollout, so multi-step consistency of the state prediction is trained, not only one-step prediction.
- **Loss:** Gaussian NLL of the next state (over real and imagined steps) + class-weighted BCE for infiltration + class-weighted CE for stage (weights 1 / 1 / 0.5). **The infiltration and stage losses are applied to the real history steps only** (`rollout_task_loss: false`); the first version also applied them to the imagined steps, trying to predict attack labels from the model's own imagined states. Those future labels are close to unpredictable here, and removing that term raised detection PR-AUC from 0.22 to 0.30 in the benchmark. Early stopping on validation loss (patience 12). Seeds fixed (`torch.use_deterministic_algorithms`); repeated runs gave identical metrics.
- Transitions come from the dataset's own timeline: consecutive windows inside one capture day (segments never span the overnight gaps).

## ATT&CK stage mapping (as in the config)

| stage | labels |
|---|---|
| Reconnaissance | PortScan |
| Initial Access | FTP-Patator, SSH-Patator, Web Brute Force, XSS, SQL Injection, Heartbleed |
| Lateral Movement | Infiltration |
| Command & Control | Bot |
| Exfiltration | **no label in the data** |
| Impact (outside the 5 stages) | DoS Hulk / GoldenEye / slowloris / Slowhttptest, DDoS |

Labels that do not fit: DoS/DDoS are Impact, so they get a sixth class rather than a wrong stage. **Exfiltration has zero training examples**, so that output can never be learned. Infiltration (36 flows), SQL Injection (21) and Heartbleed (11) never make up 5% of a window, so no window carries those labels and they are unused in practice. A window's stage is the most common attack label among its attack flows, so mixed windows get one label.

## Sanity evaluation (single split; NOT the benchmark)

Train Mon-Wed (1,450 samples: benign, Patator, DoS), validate Thu (475; web attacks), test Fri (473; Bot, DDoS, PortScan). Every attack type in the test day is unseen in training, so this mixes "temporal model quality" with "unseen attack generalisation". The benchmark ([BENCHMARK.md](BENCHMARK.md)) separates them with blocked CV and leave-one-attack-type-out.

**Dynamics** (mean squared error in normalised units, lower is better, test day):

| k (windows ahead) | world model | persistence (x_t) | training mean | linear AR(1) |
|---|---|---|---|---|
| 1 | 1.179 | 1.562 | 1.637 | 1.180 |
| 2 | **1.273** | 1.747 | 1.635 | 1.285 |
| 3 | 1.448 | 1.915 | 1.632 | **1.346** |
| 6 | 1.717 | 2.263 | 1.635 | **1.463** |

The world model now ties the linear AR(1) predictor at one step and is marginally better at two, but is worse again from three steps ahead. On the validation day it is still worse than AR(1) at every horizon (1.43 vs 1.23 at k=1, and 1.40 for the training-mean predictor), so the evidence that it has learned dynamics beyond mean reversion is weak.

**Infiltration probability** (test day, attack rate 15.2%):

| k | 0 (current) | 1 | 3 | 6 |
|---|---|---|---|---|
| ROC-AUC | 0.579 | 0.580 | 0.509 | 0.487 |
| PR-AUC | 0.466 | 0.433 | 0.257 | 0.204 |

**Stage head:** on test-day attack windows, accuracy 28%. It never predicts Reconnaissance or Command & Control, because those stages do not occur in the training days; it only gets DDoS (Impact) right.

## What this says

1. **The learned dynamics are about as good as a linear AR(1) predictor, not clearly better.** The world model beats naive persistence at every horizon. Against AR(1) it ties at one step on the test day and loses further ahead and on the validation day. At best it has learned mean reversion plus a little. The training NLL keeps falling while the validation NLL rises after a few epochs.
2. **Infiltration probability on this single split is near chance** (ROC-AUC about 0.58 at k=0 and below 0.52 from k=3), consistent with the earlier finding that there is no precursor signal in CIC-IDS2017 at 1-minute resolution. The blocked-CV benchmark gives the better estimate.
3. **The objective change is the main improvement.** Compared with the original objective, one-step dynamics error on the test day fell from 1.287 to 1.179 and validation ROC-AUC at k=0 rose from 0.746 to 0.779, because the dynamics no longer share a network with a noisy future-label task. Test ROC-AUC at k=0 is unchanged (0.587 vs 0.579). Selection of this objective was by validation score, not test.
4. **The model with the full-data weights was trained for only 4 epochs** (the best epoch found on the holdout run). It is usable for the demo pipeline, but it is not a strong model.

## Reproduce

```bash
python -W ignore src/wm_train.py                 # holdout run -> models/world_model.pt + metrics + train log
python -W ignore src/wm_train.py --mode full     # demo weights
python -W ignore src/wm_train.py --exclude-type DDoS --tag noDDoS   # leave an attack type out of training
python -W ignore src/wm_train.py --set rollout_task_loss=true --tag original   # the original objective
```

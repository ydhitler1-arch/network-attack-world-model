# World model (Phase 2)

Code: `src/world_model.py` (model, rollout, inference), `src/wm_train.py` (training, sanity evaluation), config `configs/world_model.json` (seed 42, CPU, all hyperparameters, stage mapping). Weights: `models/world_model.pt` (trained on Mon-Wed, early-stopped on Thu), `models/world_model_full.pt` (all days, for the demo only).

## Design

- Input: normalised 43-feature state S_t per 60 s window (Phase 1). Lookback L = 6 windows, rollout horizon K = 6.
- One LSTM cell (128 hidden, 100,702 parameters in total). After consuming a state it outputs:
  1. **dynamics**: a diagonal Gaussian over the next state, `N(x + delta(h), sigma(h)^2)`
  2. **infiltration probability** for the window it just consumed
  3. **ATT&CK stage** (7 classes) for that window
- **K-step rollout:** after the 6 real windows, the predicted mean state is fed back as the next input, 6 times. The infiltration and stage heads are read at every step, giving a probability series for the current window (k=0) and the next 6 (k=1..6), each read from an imagined state. Training unrolls the same rollout, so multi-step consistency is trained, not only one-step prediction.
- Loss: Gaussian NLL of the next state + class-weighted BCE for infiltration + class-weighted CE for stage (weights 1 / 1 / 0.5). Early stopping on validation loss (patience 12). Seeds fixed (`torch.use_deterministic_algorithms`); two runs gave identical metrics.
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

Train Mon-Wed (1,450 samples: benign, Patator, DoS), validate Thu (475; web attacks), test Fri (473; Bot, DDoS, PortScan). Every attack type in the test day is unseen in training, so this mixes "temporal model quality" with "unseen attack generalisation". Phase 3 separates them with blocked CV and leave-one-attack-type-out.

**Dynamics** (mean squared error in normalised units, lower is better, test day):

| k (windows ahead) | world model | persistence (x_t) | training mean | linear AR(1) |
|---|---|---|---|---|
| 1 | 1.287 | 1.562 | 1.637 | **1.180** |
| 3 | 1.518 | 1.915 | 1.632 | **1.346** |
| 6 | 1.857 | 2.263 | 1.635 | **1.463** |

On the validation day the world model (1.56 at k=1) is also worse than the training-mean predictor (1.40) and AR(1) (1.23).

**Infiltration probability** (test day, attack rate 15.2%):

| k | 0 (current) | 1 | 3 | 6 |
|---|---|---|---|---|
| ROC-AUC | 0.587 | 0.560 | 0.492 | 0.477 |
| PR-AUC | 0.456 | 0.391 | 0.299 | 0.234 |

**Stage head:** on test-day attack windows, accuracy 28%. It never predicts Reconnaissance or Command & Control, because those stages do not occur in the training days; it only gets DDoS (Impact) right.

## What this says

1. **The learned dynamics are not better than a linear AR(1) predictor.** The world model beats naive persistence but loses to AR(1) at every horizon on both the validation and test day. The training NLL keeps falling while the validation NLL rises after about epoch 5: with about 1,450 heavily overlapping samples the recurrent model overfits the dynamics. At best it has learned mean reversion. (A later experiment, [OVERFITTING_TEST.md](OVERFITTING_TEST.md), found that overfitting does not explain the world model's weaker *detection* performance.)
2. **Infiltration probability is close to chance on this split**, and gets worse further ahead (AUC below 0.5 at k >= 3), consistent with the earlier finding that there is no precursor signal in CIC-IDS2017 at 1-minute resolution.
3. **Sensitivity.** I tried two more regularised configs (hidden 32 / dropout 0.3 / wd 0.01 / lr 1e-3, and hidden 16 / dropout 0.3 / wd 0.05 / lr 5e-4; files with `_A` and `_B` suffixes). Neither beat AR(1) on dynamics. The config shipped is the default, chosen because it had the best validation AUC; I did not select on test. Test AUC at k=0 varied from 0.59 to 0.86 across the three configs, so the test estimate is noisy (one split, three attack types) and should not be quoted as a performance claim.
4. **The model with the full-data weights was trained for only 3 epochs** (the best epoch found on the holdout run). It is usable for the demo pipeline, but it is not a strong model.

## Reproduce

```bash
python -W ignore src/wm_train.py                 # holdout run -> models/world_model.pt + metrics + train log
python -W ignore src/wm_train.py --mode full     # demo weights
python -W ignore src/wm_train.py --exclude-type DDoS --tag noDDoS   # leave an attack type out of training
```

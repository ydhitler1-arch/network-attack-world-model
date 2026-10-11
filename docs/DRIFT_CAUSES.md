# Why do the neural models' thresholds drift?

[THRESHOLD_DRIFT.md](THRESHOLD_DRIFT.md) measured the problem: a threshold set for 2% false alarms gave 7.8% (LSTM classifier) and 6.5% (world model) on the test folds, up to 17% on Thursday, while logistic regression held at 2.0%. This page tests candidate causes one at a time. Code: `src/drift_analysis.py`, `src/calibration_bias.py`, `src/threshold_drift.py`. Results: `models/drift_analysis.json`, `models/calibration_bias.json`, `models/threshold_drift.json`.

## Summary

No single cause. Three explanations were rejected or shown to be minor; two are real contributors:

| hypothesis | test | verdict |
|---|---|---|
| **A. Benign traffic simply looks different on the bad days** | per-feature shift of each day's benign windows; day-vs-rest classifier | **Mostly no.** Thursday's benign traffic is nearly indistinguishable from other days |
| **B. "False alarms" are attack echoes** (the 6-window lookback still contains the attack) | split benign windows by whether an attack is in the lookback | **Minor.** About 8% of benign windows; 11% to 15% of false alarms |
| **C. The threshold is calibrated on the data used for early stopping** | controlled experiment: stop early on half the validation data, calibrate on the other half | **No.** Calibrating on unseen data is not better, slightly worse |
| **D. The threshold rests on a handful of heavy-tailed benign scores** | measure tail size; realised rate per fold; conservative finite-sample thresholds | **Yes, a real contributor** |
| **E. Unusual-volume windows get extreme scores from the neural models** | false-alarm rate by traffic-volume quintile | **Yes, where the false alarms concentrate**; not fixed by volume-binned thresholds, input clipping or dropping the volume features ([VOLUME_OOD.md](VOLUME_OOD.md)) |

## A. Does benign traffic look different on the bad days? Mostly no

Features were log-scaled and standardised over all benign windows. For each day, the standardised mean difference (SMD) of its benign windows against the other days' benign windows, for scale against the attack signal itself (attack against benign, median |SMD| **0.27**, largest `max_ports_per_src` 0.84).

| day | benign windows | median \|SMD\| vs other days | AUC of a classifier telling the day from the rest (blocked CV) | most shifted features |
|---|---|---|---|---|
| Mon | 486 | 0.17 | 0.73 | unique_dst_ips +0.6, std_log_init_win_fwd +0.5 |
| Tue | 376 | 0.15 | **0.79** | std_avg_pkt_size +0.6, mean_down_up_ratio +0.5 |
| Wed | 448 | 0.12 | 0.56 | byte_ratio_asymmetry -0.3 |
| Thu | 467 | 0.12 | **0.58** | std_log_init_win_fwd -0.5 |
| Fri | 412 | 0.12 | 0.60 | fin_ratio -0.4 |

Day-to-day shifts are about half the size of the attack signal, which is not small, but Thursday, the day with the worst false-alarm rate (14% to 17%), is almost indistinguishable from the rest. On Thursday the features that separate false-alarm windows from quiet windows (byte rate, bytes per flow, byte asymmetry, SMD about -1) have no overlap with the features that shift between days. On Tuesday there is overlap (average packet size features), so a genuine day shift plausibly contributes there.

## B. Are the "false alarms" attack echoes? Minor

For each benign test window: does the 6-window lookback contain an attack window? 157 of 1,865 benign test windows do.

| model | FPR, no attack in lookback | FPR, attack in lookback | share of all false alarms that are echoes |
|---|---|---|---|
| LSTM classifier | 7.3% | 13.8% | 15% |
| World model | 6.3% | 8.7% | 11% |
| Logistic regression (single window, no lookback) | 2.2% | 0.0% | 0% |

The echo raises the rate on those windows, but the overshoot is there without it: on Thursday, windows with no attack in the lookback still alarm at 14.4% (classifier) and 12.7% (world model), and on **Monday, which has no attacks at all**, at 6.1% and 6.3%.

## C. Is the threshold biased by calibrating on the early-stopping data? No

The neural models stop training when validation loss stops improving, so they are tuned to the validation windows; a threshold set on the same windows could be too lenient. Controlled test on the same samples (validation blocks split into two halves, both orientations, 3 seeds):

| variant | LSTM classifier FPR | world model FPR | logistic regression FPR |
|---|---|---|---|
| A. early-stop on all validation, calibrate on all (the benchmark) | 7.8% | 6.5% | 2.0% |
| C. early-stop on half 1, calibrate on the same half 1 | 8.0% | 7.1% | 3.5% |
| B. early-stop on half 1, calibrate on the **other** half | 9.3% | 9.1% | 3.5% |

If the hypothesis were right, B would be near 2%. It is not: it is slightly higher than A (paired: +1.5 points [+0.2, +2.9] for the classifier, +2.6 [+1.2, +4.1] for the world model). Logistic regression has no early stopping and still goes from 2.0% to 3.5% when its calibration sample is halved, which points to D.

## D. A handful of heavy-tailed scores sets the threshold: yes

Each fold calibrates on about 288 benign validation windows, so the 98th percentile rests on about **6 windows** above the threshold. For heavy-tailed score distributions that is noisy and biased towards overshoot (the noise can raise the realised rate far more than it can lower it below zero). Measurements:

- Realised false-alarm rate by fold: logistic regression 2.0 / 3.4 / 1.9 / 0.3 / 2.1%; LSTM classifier **4.6 / 17.6 / 0.0 / 15.3 / 2.6%**; world model 2.7 / 15.1 / 0.0 / 14.2 / 1.7%. Two folds explode, one has none: that is not a steady bias.
- Between validation and test blocks the median benign score barely moves (+0.07 inter-quartile ranges) but the 90th percentile moves **+0.27 (classifier) and +0.31 (world model)** against +0.01 for logistic regression: the neural models' tails differ between blocks.
- A threshold chosen to be exceeded by the true rate with probability at most 20% (or 10%) instead of the median (`conf80`, `conf90` in `threshold_drift.py`; validated by simulation in `tests/test_threshold_methods.py`) lowers the classifier from 7.8% to 5.9% and 5.1% (paired -2.0 [-2.9, -1.1] and -2.7 [-4.1, -1.5] points) at a cost of 0.03 to 0.04 recall; the world model goes 6.5% to 5.3% and 4.3%. The worst day stays at 10% to 14%.

## E. Where the false alarms concentrate: unusual-volume windows

False-alarm rate on benign windows by traffic volume (quintiles of flows per minute; the lowest fifth is under about 320 flows, the highest over about 1,400):

| model | lowest | 2nd | 3rd | 4th | highest |
|---|---|---|---|---|---|
| Logistic regression | 1.3% | 0.8% | 0.3% | 1.1% | 6.4% |
| LSTM classifier | **15.8%** | 4.6% | 4.1% | 5.5% | 9.2% |
| World model | **13.1%** | 3.8% | 3.3% | 4.7% | 7.6% |
| LSTM classifier, Thursday only | **40%** | 1% | 5% | 14% | 25% |
| World model, Thursday only | **35%** | 0% | 3% | 10% | 20% |

The neural models raise alarms on the quietest and the busiest benign minutes, and Thursday has many of them; logistic regression is only affected at the busy end. Separate thresholds per volume bin (2 or 3 bins) did **not** fix the neural models (8.4% false alarms, no better than a single threshold), so this is not just a calibration mismatch between volume levels, and the likely explanation is that those windows are out of distribution for the networks (rare in training, ratio features such as concentration shares are noisy when a window has few flows). A follow-up ([VOLUME_OOD.md](VOLUME_OOD.md)) tested it by clipping inputs more tightly and by dropping the volume-tracking features: clipping did nothing and dropping the features helped only a little, so extreme input values are not the main mechanism.

## What helps, ranked

1. **Delayed analyst feedback on benign windows** roughly halves the excess false alarms without losing recall ([THRESHOLD_DRIFT.md](THRESHOLD_DRIFT.md)).
2. **A conservative finite-sample threshold** removes about a third of the excess for a small recall cost.
3. **Using the simpler model.** Logistic regression's threshold is stable (2.0%) and it scores better.

## Caveats

- Everything is one dataset, one lab network and 41 attack episodes. The day-level false-alarm rates rest on a few hundred benign windows each.
- Hypothesis testing was sequential and exploratory: five hypotheses, the volume finding came after looking at the false-alarm drivers. Treat E as a lead, not a result.
- The score-distribution diagnostics use the cached scores of one training run per seed (3 seeds).
- The false-alarm driver analysis compares windows within a day and is descriptive; it does not show that those features cause the alarms.

## Reproduce

```bash
python -W ignore src/threshold_drift.py            # scores for all models, ~4 min (caches in models/bench_drift_scores.pkl)
python -W ignore src/drift_analysis.py             # diagnostics A, B, D, E, ~1 min
python -W ignore src/calibration_bias.py           # control experiment C, ~8 min
```

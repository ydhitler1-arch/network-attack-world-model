# Threshold drift: measuring it, and what fixes it

Code: `src/threshold_drift.py` (about 4 min; `--reuse` re-evaluates from cached scores). Results: `models/threshold_drift.json`. No-look-ahead checks: `tests/test_leakage.py`.

## The problem

The benchmark calibrates an alert threshold once, on a validation fold, for a 2% false-alarm rate (FPR) on benign windows, and then applies it to the test fold. On the test days the realised FPR ranged from 2% to 8% depending on the model ([BENCHMARK.md](BENCHMARK.md)). In an intrusion-detection setting that matters more than a few points of PR-AUC: a detector whose false-alarm rate is three times its target gets switched off.

## Setup

Same blocked-CV samples as the main benchmark (2,095 test samples, 11% attack), task: detect the current window, neural models averaged over 3 seeds, world model with the revised objective. Four ways of setting the threshold, evaluated on identical samples:

| strategy | what it does | needs |
|---|---|---|
| **fixed** | the benchmark's method: threshold = 98th percentile of benign validation scores | labels at calibration time only |
| **recenter** | subtract the median of the previous 60 scores from every score; calibrate on the re-centred validation stream | no labels |
| **feedback** | start from the fixed threshold; whenever a window is confirmed benign (labels arrive 5 windows late), nudge the threshold by online quantile tracking, `threshold += step * (alarmed - 0.02)`, step = 0.05 benign-validation standard deviations | an analyst confirming benign windows |
| **oracle** (reference only) | per-day threshold at the target rate using that day's own benign test scores | not deployable |

The step size and delay of `feedback` were fixed before the first run and are not tuned; a sensitivity sweep is shown below.

## Results

Realised false-alarm rate (target 2%) and recall, pooled over the test samples, 95% intervals from a block bootstrap. The days are Monday to Friday (Monday has no attacks).

| model | strategy | FPR [95% CI] | recall [95% CI] | F1 | FPR by day (Mon / Tue / Wed / Thu / Fri) |
|---|---|---|---|---|---|
| Logistic regression | fixed | 2.0% [0.9, 3.3] | 0.32 [0.15, 0.54] | 0.43 | 2.4 / 0.9 / 0.8 / 4.0 / 1.4 |
| | recenter | 2.3% [1.3, 3.5] | 0.32 [0.17, 0.49] | 0.43 | 2.6 / 1.3 / 1.3 / 4.5 / 1.1 |
| | feedback | 2.3% [1.1, 3.5] | 0.33 [0.16, 0.53] | 0.43 | 2.2 / 0.9 / 1.3 / 5.0 / 1.4 |
| | oracle | 2.8% [2.3, 3.2] | 0.41 [0.23, 0.59] | 0.50 | 3.1 / 3.1 / 2.6 / 2.5 / 2.5 |
| LSTM classifier | fixed | **7.8%** [3.4, 13.2] | 0.30 [0.08, 0.56] | 0.31 | 6.1 / 11.2 / 2.8 / **17.0** / 1.9 |
| | recenter | 6.3% [3.0, 10.7] | 0.21 [0.05, 0.42] | 0.25 | 2.9 / 7.9 / 4.0 / 15.4 / 1.5 |
| | feedback | **4.4%** [1.7, 7.7] | 0.29 [0.07, 0.54] | 0.35 | 4.7 / 5.0 / 1.6 / 9.5 / 0.8 |
| | oracle | 2.8% [2.3, 3.3] | 0.24 [0.08, 0.43] | 0.33 | 3.1 / 3.1 / 2.6 / 2.5 / 2.5 |
| World model | fixed | **6.5%** [2.8, 11.2] | 0.27 [0.08, 0.49] | 0.30 | 6.3 / 9.4 / 1.8 / **13.8** / 1.1 |
| | recenter | 5.5% [2.4, 9.6] | 0.19 [0.04, 0.38] | 0.23 | 2.8 / 7.4 / 2.9 / 13.0 / 1.2 |
| | feedback | **3.8%** [1.5, 6.6] | 0.25 [0.06, 0.48] | 0.32 | 5.1 / 4.1 / 1.3 / 7.4 / 0.7 |
| | oracle | 2.8% [2.3, 3.3] | 0.26 [0.10, 0.46] | 0.35 | 3.1 / 3.1 / 2.6 / 2.5 / 2.5 |

(The oracle lands at 2.8% rather than 2.0% because its per-day threshold is fitted on small groups of benign windows, a few dozen per fold and day; treat it as "about the target".)

**Paired comparison against the fixed threshold** (same resampled blocks, so these are the numbers to trust, not the overlap of the intervals above):

| model | strategy | change in FPR | change in recall |
|---|---|---|---|
| LSTM classifier | feedback | **-3.4 points [-6.5, -1.3]** | -0.012 [-0.049, +0.008] |
| | recenter | -1.5 points [-3.9, +0.7] | -0.086 [-0.235, +0.000] |
| World model | feedback | **-2.7 points [-5.6, -0.7]** | -0.014 [-0.046, +0.000] |
| | recenter | -1.0 points [-3.1, +0.9] | -0.077 [-0.227, +0.009] |
| Logistic regression | feedback | +0.3 points [-0.1, +0.6] | +0.005 [-0.010, +0.023] |
| | recenter | +0.3 points [-0.3, +0.9] | -0.002 [-0.083, +0.069] |

## What this says

1. **The drift is real, concentrated on specific days, and specific to the neural models.** Logistic regression's threshold holds (2.0% overall). The LSTM classifier and the world model overshoot badly on Thursday (17% and 14%) and Tuesday (11% and 9%), and are fine on Wednesday and Friday. Their score distributions shift between days; the linear model's do not.
2. **Label-free re-centring does not work.** It lowers false alarms slightly (not significantly) and costs about 8 points of recall, so F1 gets worse. The trailing median is dragged upward during a sustained attack, which hides the attack. This is the standard weakness of adapting to "recent traffic" without labels.
3. **Analyst feedback on benign windows does work for the neural models.** It cuts the overshoot by about half (LSTM 7.8% to 4.4%, world model 6.5% to 3.8%, both significant in the paired test), cuts the worst day roughly in half (17% to 9.5%, 13.8% to 7.4%), and leaves recall essentially unchanged. It does nothing for logistic regression, which did not need it.
4. **It does not reach the target.** After feedback the neural models are still at about 4%, twice the 2% target, and Thursday is still at 7% to 10%.
5. **The simplest fix is the model.** Logistic regression with a fixed threshold is better calibrated than either neural model with feedback. If an operator cannot supply feedback, the stable model is the practical choice.

## Sensitivity of the feedback method (nothing tuned; every cell reported)

FPR / recall / worst-day FPR, pooled; the settings used above are in bold.

| model | step 0.02, delay 5 | step 0.05, delay 5 | step 0.1, delay 5 | step 0.05, delay 30 | step 0.1, delay 30 |
|---|---|---|---|---|---|
| LSTM classifier | 5.8% / 0.29 / 12.6% | **4.4% / 0.29 / 9.5%** | 3.4% / 0.29 / 7.6% | 5.0% / 0.29 / 11.6% | 4.2% / 0.29 / 10.3% |
| World model | 4.9% / 0.26 / 10.0% | **3.8% / 0.25 / 7.4%** | 3.0% / 0.26 / 6.2% | 4.3% / 0.25 / 8.8% | 3.5% / 0.26 / 8.0% |
| Logistic regression | 2.0% / 0.32 / 4.0% | **2.3% / 0.33 / 5.0%** | 2.4% / 0.33 / 5.5% | 2.1% / 0.33 / 4.3% | 2.4% / 0.33 / 5.0% |

Larger steps and shorter delays help the neural models: at step 0.1 and delay 5 the world model reaches 3.0% and the classifier 3.4%, with recall unchanged. The setting I fixed in advance was therefore not the best, and the optimum is unknown because larger steps were not tried. A very large step would simply chase the latest windows, so there is a limit that this sweep does not locate.

## Caveats

- **Feedback needs an analyst.** The method assumes someone confirms windows as benign within a few minutes. Windows nobody reviews give no feedback, and a real analyst's labels are noisier than the ground truth used here.
- **Small samples.** About 1,865 benign and 230 attack test samples; the day-level false-alarm rates rest on a few hundred benign windows each, so single-day numbers are rough (about +/- 1 point at 2%, more at 10%).
- **Three seeds, one dataset, one lab network.** Whether other datasets show the same day-to-day shifts is untested.
- The intervals in the main table are unpaired and overlap heavily; the paired table is the one that supports the conclusions.
- `tests/test_leakage.py` checks that neither adaptive method uses a score or label from the future.

## Reproduce

```bash
python -W ignore src/threshold_drift.py            # ~4 min, caches scores in models/bench_drift_scores.pkl (git-ignored)
python -W ignore src/threshold_drift.py --reuse    # re-evaluate from the cache (~1 min)
```

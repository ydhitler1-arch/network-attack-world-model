# Can the strongest model be improved?

Logistic regression on the current window is the best detector in [BENCHMARK.md](BENCHMARK.md) (PR-AUC 0.451). This page tries to beat it with the obvious options. **Result: nothing does, and the way the options fail is informative.** Code: `src/improve_baseline.py`, `src/vol_threshold_curve.py`. Results: `models/improve_baseline.json`, `models/vol_threshold_curve.json`.

## Protocol

Same blocked-CV samples, features and test folds as the benchmark (2,095 test samples, 11% attack). [DRIFT_CAUSES.md](DRIFT_CAUSES.md) showed that a threshold calibrated on the data used for model selection is unreliable, so every candidate, including the baseline, follows one clean protocol:

1. split the validation blocks into two halves;
2. select hyper-parameters (and early-stop, for neural models) on half 1;
3. calibrate the alert threshold (2% false alarms on benign windows) on half 2;
4. evaluate on the test fold; both orientations (halves swapped) are averaged.

Nothing is selected on test. All candidates are reported, with paired block-bootstrap comparisons against the baseline. With about 15 candidates, one "significant" difference would be weak evidence by itself (none turned up in the favourable direction).

## Candidates

- **lr_base**: the benchmark's logistic regression (C = 1, balanced classes), the baseline.
- **lr_tuned**: logistic regression with C in {0.01 ... 3} and class weighting chosen on the selection half; **lr_l1**: L1 penalty; **lr_poly2**: pairwise interactions among the 12 most informative features.
- **xgb_cur**, **xgb_seq**: gradient-boosted trees on the current window (43 features) or on a 6-window summary (172 features), depth and tree count chosen on the selection half; **extra_trees**.
- **ens_***: equal-weight averages of z-scored scores: logistic regression with the LSTM classifier, the world model, both, or XGBoost.
- **lstm_clf**, **world_model**: the neural models under the same protocol, for reference.
- **+vol2**: the same scores with one threshold per traffic-volume half (split at the median flow count of the benign calibration windows).

## Results

PR-AUC is threshold-free; precision, recall, F1 and false-alarm rate (FPR) use the calibration-half threshold. "Selection half" is the PR-AUC on the half used for choosing among settings.

| candidate | PR-AUC [95% CI] | F1 | recall | FPR | selection-half PR-AUC | PR-AUC minus baseline [95% CI] |
|---|---|---|---|---|---|---|
| **lr_base** (baseline) | **0.451** [0.23, 0.67] | 0.42 | 0.35 | 3.5% | 0.365 | |
| lr_l1 | 0.393 [0.18, 0.62] | 0.37 | 0.32 | 4.7% | 0.395 | -0.054 [-0.13, +0.03] |
| ens_lr+xgb | 0.382 [0.18, 0.58] | 0.38 | 0.35 | 6.1% | | -0.063 [-0.20, +0.06] |
| ens_lr+clf | 0.377 [0.17, 0.62] | 0.40 | 0.39 | 6.5% | | -0.071 [-0.23, +0.06] |
| lr_tuned | 0.366 [0.15, 0.59] | 0.36 | 0.29 | 4.1% | 0.425 | -0.082 [-0.18, +0.01] |
| ens_lr+wm | 0.364 [0.16, 0.60] | 0.40 | 0.39 | 6.5% | | -0.084 [-0.24, +0.05] |
| ens_lr+clf+wm | 0.341 [0.13, 0.58] | 0.37 | 0.37 | 7.7% | | -0.106 [-0.30, +0.06] |
| xgb_cur | 0.284 [0.13, 0.48] | 0.31 | 0.27 | 6.2% | 0.391 | -0.152 [-0.37, +0.03] |
| lstm_clf | 0.279 [0.12, 0.47] | 0.27 | 0.27 | 9.3% | | -0.164 [-0.35, -0.00] |
| world_model | 0.275 [0.11, 0.50] | 0.29 | 0.29 | 9.1% | | -0.166 [-0.36, +0.01] |
| xgb_seq | 0.265 [0.12, 0.48] | 0.35 | 0.34 | 7.6% | 0.455 | -0.172 [-0.36, -0.00] |
| extra_trees | 0.263 [0.11, 0.48] | 0.37 | 0.32 | 5.3% | 0.313 | -0.172 [-0.36, +0.03] |
| lr_poly2 | 0.218 [0.08, 0.39] | 0.16 | 0.12 | 4.9% | 0.284 | -0.225 [-0.37, -0.06] |

## What this says

1. **No candidate beats the baseline** on PR-AUC, and every one is below it in point estimate. One is clearly worse in the paired test (the interaction model, -0.225 [-0.37, -0.06]); two more only just exclude zero (XGBoost on summaries and the LSTM classifier); the rest cannot be separated from it.
2. **Model selection on a small validation half is unreliable here.** On the selection half the complex models look best (XGBoost on summaries 0.455, tuned logistic regression 0.425, against 0.365 for the baseline); on test they are worse (0.265, 0.366, 0.451). With only about 150 selection windows, picking the best of 12 to 16 settings fits noise. The baseline's default (C = 1) was fixed before any of this, and it holds up.
3. **Ensembling does not help.** Averaging logistic regression with the neural models or XGBoost scores below logistic regression alone; the weaker members dilute it.
4. **The neural models need the validation data for early stopping.** With half the validation data they score 0.28 and 0.275 here, against 0.350 and 0.299 in the main benchmark. The comparison with logistic regression is therefore a little harsher on them under this protocol.
5. **Calibration sample size shows up in the false-alarm rate.** The baseline's realised rate is 3.5% here against 2.0% in the benchmark, only because the calibration half is half as large (see [DRIFT_CAUSES.md](DRIFT_CAUSES.md), part D).

## One promising lead: volume-conditional thresholds

Keeping the baseline's scores but setting a separate threshold for low-volume and high-volume minutes raised F1 from 0.42 to **0.52** (recall 0.35 to 0.48), but the false-alarm rate also rose (3.5% to 4.4%), so single operating points are not comparable. The fair comparison sweeps the target false-alarm rate for both detectors and compares recall at matched realised false-alarm rates (`src/vol_threshold_curve.py`, calibration on the held-out half, block bootstrap):

| realised false-alarm rate | recall, one global threshold | recall, one threshold per volume half | paired difference [95% CI] | volume-conditional better in |
|---|---|---|---|---|
| 2% | 0.276 [0.13, 0.50] | 0.335 [0.18, 0.53] | +0.046 [-0.013, +0.099] | 94% of resamples |
| 3% | 0.337 [0.17, 0.54] | 0.360 [0.19, 0.57] | +0.038 [-0.017, +0.116] | 88% |
| 5% | 0.414 [0.23, 0.62] | 0.503 [0.29, 0.69] | +0.075 [-0.004, +0.193] | 96% |

At matched false-alarm rates the volume-conditional version has the higher recall at every level (+0.04 to +0.08) and in 88% to 96% of resamples, but **every paired interval includes zero**. It is consistent and plausible (attacks and benign traffic have different score levels at different volumes), and worth testing on larger data; it is not established here. It was found by looking at the drift analysis and tried on the same folds, so it carries forking-paths optimism. The same idea did not help the neural models.

## Conclusion

On this data and these features, the best available detector is the plain logistic regression, and tuning, interactions, boosted trees, forests and ensembles all fail to improve it under an honest protocol. Gains are more likely to come from more data (more attack episodes, a finer time resolution) than from more model.

## Caveats

- 41 attack episodes and 5 capture days: wide intervals; differences among the weaker candidates are within noise.
- Hyper-parameter grids were small and fixed in advance; larger or smarter searches could find a better setting, but the selection noise above argues against expecting much.
- The neural models were not re-tuned for this protocol.

## Reproduce

```bash
python -W ignore src/calibration_bias.py        # scores for the neural models on the two validation halves (~8 min)
python -W ignore src/improve_baseline.py        # ~4 min
python -W ignore src/vol_threshold_curve.py     # ~1 min
```

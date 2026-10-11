# Are unusual-volume windows out of distribution for the neural models?

[DRIFT_CAUSES.md](DRIFT_CAUSES.md) (part E) found that the neural models' false alarms concentrate on the quietest and the busiest benign minutes, and offered an untested explanation: those windows are out of distribution (OOD), with extreme standardised feature values. This page tests it. Code: `src/volume_ood_test.py`. Results: `models/volume_ood_test.json`.

## Design

If extreme inputs on unusual-volume windows cause the extreme scores, interventions that bound or remove the volume signal should cut those false alarms without hurting detection. Same blocked-CV samples, protocol and thresholds as `src/threshold_drift.py` (threshold at the 98th percentile of benign validation scores), 3 seeds; the baseline is that experiment's cached scores.

| intervention | what it does |
|---|---|
| `clip3`, `clip2` | clip standardised inputs to +/-3 or +/-2 instead of +/-8 |
| `drop_vol` | drop the features that track traffic volume: those with |Spearman| > 0.6 against log flow count among the fold's benign **training** windows (7 features in every fold except one with 8: `flow_count`, `packet_rate`, `byte_rate`, `unique_dst_ips`, `unique_dst_ports`, `unique_src_ips`, `max_dsts_per_src`) |
| `clip3+drop_vol` | both |

Logistic regression is a control: its false-alarm rate was already stable. There are no empty (zero-flow) gap-fill windows in the data, so that explanation was ruled out beforehand.

## Results

False-alarm rate (FPR, target 2%), worst single day, FPR on the lowest-volume fifth of benign windows, and detection quality. "Paired" columns are changes against the baseline on the same resampled blocks.

| model | variant | FPR | worst day | FPR, lowest-volume fifth | PR-AUC | recall | paired change in FPR [95% CI] |
|---|---|---|---|---|---|---|---|
| LSTM classifier | baseline | 7.8% | 17.0% | 15.6% | 0.350 | 0.30 | |
| | clip3 | 7.5% | 16.5% | 14.8% | 0.327 | 0.29 | -0.3 points [-0.8, +0.2] |
| | clip2 | 7.6% | 15.1% | 14.5% | 0.328 | 0.30 | -0.2 [-1.2, +1.0] |
| | drop_vol | 6.4% | 12.9% | 12.6% | 0.347 | 0.33 | **-1.4 [-3.2, -0.0]** |
| | clip3+drop_vol | 6.5% | 11.8% | 12.6% | 0.354 | 0.34 | -1.3 [-3.2, +0.3] |
| World model | baseline | 6.5% | 13.8% | 13.1% | 0.299 | 0.27 | |
| | clip3 | 6.1% | 12.8% | 11.8% | 0.298 | 0.26 | -0.4 [-1.2, +0.2] |
| | clip2 | 6.4% | 12.4% | 11.5% | 0.298 | 0.26 | -0.1 [-1.1, +1.0] |
| | drop_vol | 5.6% | 12.2% | 9.5% | 0.331 | 0.30 | -1.0 [-2.7, +0.4] |
| | clip3+drop_vol | 7.1% | 13.0% | 12.4% | **0.369** | 0.33 | +0.6 [-1.1, +1.9] |
| Logistic regression (control) | baseline | 2.0% | 4.0% | 1.3% | 0.451 | 0.32 | |
| | drop_vol | 2.3% | 5.5% | 1.3% | 0.456 | 0.30 | +0.3 [-0.2, +0.7] |
| | clip3+drop_vol | 2.2% | 4.3% | 1.0% | 0.419 | 0.24 | +0.2 [-0.5, +1.0] |

Paired change in **PR-AUC** against the baseline (block bootstrap, 1,000 resamples):

| model | clip3 | drop_vol | clip3+drop_vol |
|---|---|---|---|
| LSTM classifier | -0.023 [-0.055, +0.009] | -0.003 [-0.097, +0.087] | +0.003 [-0.088, +0.097] |
| World model | -0.001 [-0.020, +0.020] | +0.031 [-0.020, +0.085] | **+0.069 [+0.011, +0.122]** (better in 99% of resamples) |
| Logistic regression | -0.019 [-0.126, +0.055] | +0.005 [-0.044, +0.071] | -0.032 [-0.112, +0.022] |

## What this says

1. **The "extreme input values" explanation is not supported.** Tighter clipping changes the false-alarm rate by less than half a point, and slightly lowers the classifier's PR-AUC. Bounding the inputs does not bound the problem.
2. **Removing the volume-tracking features helps a little, not enough.** The classifier's false alarms fall from 7.8% to 6.4% (borderline significant) and its worst day from 17% to 12.9%; for the world model 6.5% to 5.6% (not significant). The lowest-volume windows still alarm at 12.6% and 9.5%, against 4% to 5% in the middle of the volume range. The volume sensitivity is therefore not only the explicit volume inputs: ratio and concentration features are also unreliable when a window has few flows, and several features correlate with volume to a lesser degree (13 features have |rho| > 0.5).
3. **Logistic regression does not need the volume features.** Without them it keeps its PR-AUC (0.456 against 0.451), so a 36-feature version is as good, which says the volume information carries little of its detection ability.
4. **A possible side benefit for the world model**, flagged as a lead: with the volume features removed and inputs clipped to +/-3 its detection PR-AUC rises from 0.299 to 0.369 (paired +0.069 [+0.011, +0.122]), with recall 0.27 to 0.33, though its false-alarm rate does not improve (7.1%). This is one significant result out of nine paired PR-AUC comparisons, found after looking at the test folds, so it is weak evidence on its own. The two interventions alone are not significant for the world model, and neither helps the classifier. It needs confirming on other data before being believed.

## Conclusion

Out-of-distribution extremes are not the main mechanism behind the volume-dependent false alarms, and the cheap fixes (clipping, dropping volume features) only dent them. The cause remains partly unexplained; what is left is that the neural models are simply less well calibrated than logistic regression on rare-volume windows, which the training data covers thinly. More data across traffic levels, or training with volume augmentation, are the untested remedies. Delayed analyst feedback ([THRESHOLD_DRIFT.md](THRESHOLD_DRIFT.md)) remains the best practical fix.

## Caveats

- Four interventions and three models were tried on the same folds, and results are described with their paired intervals; one significant result among many comparisons should not be over-read.
- 41 attack episodes; the lowest-volume quintile contains about 370 benign test windows.
- The volume-feature set was chosen by a fixed rule (|Spearman| > 0.6) on training data only; other cut-offs were not tried.
- Volume augmentation (training with synthetically thinned windows) was not tried, because the state vector is built from flow aggregates and cannot be thinned after the fact without rebuilding the windows from flows.

## Reproduce

```bash
python -W ignore src/threshold_drift.py          # baseline scores (~4 min)
python -W ignore src/volume_ood_test.py          # ~20 min; --reuse re-evaluates from cached scores
```

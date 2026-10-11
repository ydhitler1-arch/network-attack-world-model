# Does overfitting explain why the world model loses to logistic regression?

Code: `src/overfit_test.py` (about 10 min). Results: `models/overfit_test.json`, `models/objective_test.json`.

> **Update.** The labels "(shipped)" and "shipped objective" below refer to the original objective, which was the shipped one when these experiments ran. After the follow-up experiment the shipped model was retrained with the revised objective (`rollout_task_loss: false`) and the full benchmark was re-run (world model PR-AUC 0.299, which reproduces the 0.299 in the follow-up table). Original-objective results are kept in `models/v1_original_objective/`.

## Hypothesis and predictions

The benchmark found the world model (PR-AUC 0.22) worse than logistic regression (0.45). A first explanation was overfitting: about 100k parameters trained on about 1,500 heavily overlapping samples. If true:

- **P1** the neural models show a much bigger train-vs-test gap than a simple model;
- **P2** shrinking capacity, regularising, or down-weighting the dynamics loss raises test PR-AUC towards logistic regression;
- **P3** validation PR-AUC picks the good configuration without looking at test.

## Method

Same protocol and samples as the main benchmark (blocked 5-fold CV with a 12-window purge, scaler refit per fold, task: detect the current window, 2,095 test samples, 11% attack, 3 seeds). Twelve neural configurations plus logistic regression. PR-AUC is measured on a training subsample, the validation fold and the test fold. Nothing is selected on test: the "selected" configuration is the one with the best validation PR-AUC.

## Results (PR-AUC, mean of 3 seeds)

| config | parameters | train | val | test [95% CI] | test minus logistic regression [95% CI] |
|---|---|---|---|---|---|
| logistic regression | 44 | 0.865 | 0.366 | **0.451** [0.23, 0.67] | |
| world model, hidden 128 (shipped) | 100,702 | 0.835 | 0.342 | 0.219 [0.11, 0.37] | -0.222 [-0.41, -0.05] |
| world model, hidden 64 | 34,014 | 0.785 | 0.330 | 0.228 [0.12, 0.39] | -0.209 [-0.38, -0.02] |
| world model, hidden 32 | 12,958 | 0.769 | 0.264 | 0.196 [0.10, 0.38] | -0.228 [-0.41, -0.04] |
| world model, hidden 16 | 5,502 | 0.772 | 0.306 | 0.241 [0.13, 0.43] | -0.191 [-0.38, -0.00] |
| world model, hidden 8 | 2,542 | 0.556 | 0.179 | 0.248 [0.11, 0.45] | -0.192 [-0.41, -0.00] |
| world model, hidden 32, dropout 0.3, weight decay 0.05 | 12,958 | 0.772 | 0.268 | 0.211 [0.10, 0.40] | -0.218 [-0.40, -0.02] |
| world model, hidden 128, dynamics weight 0.1 | 100,702 | 0.828 | 0.320 | 0.212 [0.11, 0.38] | -0.225 [-0.40, -0.06] |
| world model, hidden 32, dynamics weight 0.1 | 12,958 | 0.790 | 0.298 | 0.234 [0.12, 0.42] | -0.198 [-0.38, -0.02] |
| LSTM classifier, hidden 128 | 100,702 | 0.895 | 0.365 | 0.350 [0.16, 0.59] | -0.101 [-0.35, +0.10] |
| LSTM classifier, hidden 32 | 12,958 | 0.881 | 0.354 | 0.399 [0.18, 0.63] | -0.057 [-0.30, +0.16] |
| LSTM classifier, hidden 16 | 5,502 | 0.853 | 0.380 | 0.379 [0.16, 0.64] | -0.067 [-0.30, +0.17] |
| LSTM classifier, hidden 8 | 2,542 | 0.692 | 0.247 | 0.278 [0.11, 0.55] | -0.154 [-0.39, +0.08] |

Selected by validation: LSTM classifier with 16 hidden units (test 0.379, logistic regression 0.451). Best world model by validation: the shipped one (test 0.219).

## What this says

1. **P1 is not specific to big models.** Logistic regression scores 0.865 on its own training data and 0.451 on test (gap 0.41), about as large as the neural models' gaps (0.31 to 0.62). A large train-test gap is partly caused by traffic changing between blocks and days, not only by capacity. (The training score is in-sample, so some gap is expected for any fitted model.)
2. **P2 fails for the world model.** Cutting 100,702 parameters to 2,542, adding dropout and weight decay, or lowering the dynamics-loss weight leaves test PR-AUC at 0.20 to 0.25. Every variant stays below logistic regression (the intervals for the two smallest touch zero at the edge). The differences among world-model variants are inside the confidence intervals.
3. **P2 holds weakly for the plain classifier.** Making it smaller helped (0.350 at 100k parameters, 0.399 at 13k, 0.379 at 5.5k), then hurt at 2.5k (0.278). Its difference from logistic regression is never significant, including at the shipped size, so this is not clear evidence either.
4. **The world model is worse than the classifier at every size** (0.219 vs 0.350, 0.196 vs 0.399, 0.241 vs 0.379, 0.248 vs 0.278 at hidden 128, 32, 16 and 8). Same network, same data; what differs is the training objective. So the loss in detection comes from how the world model is trained, not from how large it is.
5. **P3:** validation PR-AUC picked a reasonable classifier but is noisy (about 475 validation samples per fold, few attack episodes), so treat model selection on it with caution.

## Conclusion

**Overfitting through model capacity is not the main reason the world model loses.** The follow-up below points at the training objective: the world model also learned to predict attack labels at six future steps from its own imagined states. Those labels are close to unpredictable (forecasting 3 minutes ahead is at chance), so that part of the loss added noise to the detection head. Removing it recovers much of the gap to the plain classifier, but not the gap to logistic regression.

## Follow-up: is the training objective the cause?

Prediction written down before running (`src/overfit_test.py objective`, results in `models/objective_test.json`): if training the infiltration head on imagined future states is what hurts detection, a world model with those task losses removed should recover the classifier's level (test PR-AUC about 0.35 or more at the same size); if it stays near 0.2, the hypothesis is wrong. Two new training options (off by default, so the shipped weights and earlier results are unchanged): `rollout_task_loss: false` applies the infiltration and stage losses to the real history steps only (the dynamics loss still uses the rollout), and `train_horizon: 0` trains with no rollout at all.

| config | parameters | train | val | test [95% CI] | test minus logistic regression [95% CI] |
|---|---|---|---|---|---|
| logistic regression | 44 | 0.865 | 0.366 | **0.451** [0.23, 0.67] | |
| world model, hidden 128 (shipped objective) | 100,702 | 0.835 | 0.342 | 0.219 [0.11, 0.37] | -0.222 [-0.41, -0.05] |
| + task losses on real steps only | 100,702 | 0.900 | 0.383 | 0.299 [0.14, 0.53] | -0.143 [-0.35, +0.04] |
| + no rollout in training | 100,702 | 0.897 | 0.361 | 0.331 [0.15, 0.56] | -0.116 [-0.35, +0.08] |
| world model, hidden 32 (shipped objective) | 12,958 | 0.769 | 0.264 | 0.196 [0.10, 0.38] | -0.228 [-0.41, -0.04] |
| + task losses on real steps only | 12,958 | 0.882 | 0.376 | 0.329 [0.16, 0.58] | -0.112 [-0.33, +0.09] |
| + no rollout in training | 12,958 | 0.890 | 0.364 | 0.384 [0.17, 0.61] | -0.068 [-0.31, +0.15] |
| LSTM classifier, hidden 128 | 100,702 | 0.895 | 0.365 | 0.350 [0.16, 0.59] | -0.101 [-0.35, +0.10] |
| LSTM classifier, hidden 32 | 12,958 | 0.881 | 0.354 | 0.399 [0.18, 0.63] | -0.057 [-0.30, +0.16] |

**Reading it.** The prediction largely holds: removing the future-state task loss raises the world model from 0.22 and 0.20 to 0.30 and 0.33, and training with no rollout brings it to 0.33 and 0.38, close to the plain classifier at the same size (0.35 and 0.40). Validation PR-AUC rises in the same direction (0.342 to 0.383 at hidden 128). The effect is consistent in sign and size across both model sizes and both variants, so the training objective explains much of the earlier gap. Choosing by validation alone selects "real steps only, hidden 128" (test 0.299).

**What it does not show.**
- No paired significance test was run between the fixed and the shipped variants; the intervals above overlap, so the improvement is consistent but not established as significant.
- Fixing the objective brings the world model up to the classifier, **not above it**, and not above logistic regression (0.30 to 0.38 against 0.451; the differences are not significant, but logistic regression stays ahead on the point estimate).
- Training without any rollout is the best world-model variant here, so the rollout in training did not help detection.
- The follow-up was run on the original shipped model's setup; afterwards the shipped weights were retrained with the revised objective and the benchmark re-run (see the update at the top).

## Caveats

- 41 attack episodes: intervals are wide, and most pairwise differences among neural variants are not significant.
- Not tried: input-noise augmentation, a Transformer, other learning rates, different lookback, early-stopping criteria other than validation infiltration loss.
- The experiment tests the model, not the data: if traffic in this dataset has little predictive structure beyond what logistic regression captures, no model will beat it.

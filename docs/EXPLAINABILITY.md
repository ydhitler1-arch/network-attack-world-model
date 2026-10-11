# Explainability (Phase 4)

Code: `src/explain.py`. Outputs: `models/explanations_full.parquet` (per-window table), `models/explain_background.npz` (SHAP reference set), `models/explanation_validation.json` (all numbers below).

## What is explained

Every prediction the world model makes gets a SHAP explanation:

- **Infiltration probability for the current window and for each of the 6 rollout steps (k = 0..6):** attribution of the infiltration logit to each of the 43 state features, summed over the 6-window lookback, plus the split across lags (which of the last 6 minutes mattered).
- **Predicted ATT&CK stage (k = 0):** attribution of the predicted stage's logit to the 43 features.

Method: SHAP values by the expected-gradients estimator (same estimator as `shap.GradientExplainer`), implemented in `explain.py` so that it differentiates through both the history and the K-step rollout. Reference distribution: 200 benign 6-window histories (benign windows with benign history) from CIC-IDS2017. A SHAP value is "how much this feature (in this window or lag) moved the logit away from its benign-reference average". Positive = pushes towards infiltration.

The per-window table has the top 5 features per (window, horizon), with the SHAP value, the raw feature value and its z-score, and for k = -1 the top 5 drivers of the predicted stage. `describe_window(table, t, k)` turns a row group into a sentence, for example:

> P(infiltration at t+0) = 1.00; drivers: top_pair_share (+3.17, z=+6.0), large_pkt_ratio (+2.51, z=+8.0), max_ports_per_src (+2.29, z=+5.4), mean_avg_pkt_size (-1.51, z=+5.5), ...

(a DDoS window). The same code explains logistic regression with exact linear SHAP (`explain_logreg`).

## Validation (nothing here is taken on trust)

| check | result |
|---|---|
| **Completeness**: sum of SHAP vs f(x) minus reference value (256 draws per explanation) | mean absolute error 0.159 logits, 9.5% of the explained quantity for the demo model (12.9% for the holdout model; it shrinks about 1/sqrt(N)). Explanations are approximate, not exact. |
| **Agreement with the `shap` library**: `shap.GradientExplainer` on 20 windows, k = 0 | mean Pearson r = 0.99 (min 0.98) between per-feature attributions |
| **Faithfulness (deletion)**: holdout model, 100 highest-scoring Friday windows (never in training); reset the m top-SHAP features to the benign reference | logit drop for m = 3 / 5 / 10: top-SHAP **+2.14 / +2.81 / +4.11**; random features +0.09 / +0.02 / +0.21; bottom-SHAP (most negative) -2.45 / -3.19 / -4.01. The attributions pick the features that actually drive the score. |
| **Logistic regression linear SHAP** | exact (max error 8e-6) |
| **World model vs logistic regression** on attack windows of the test day | mean Spearman correlation of |SHAP| rankings 0.45: they partly agree on which features matter, not fully |

## What the explanations show

- Global importance for the infiltration logit (mean |SHAP|, k = 0): `max_ports_per_src` 0.64, `small_pkt_ratio` 0.50, `top_pair_share` 0.47, `pkt_ratio_bwd_fwd` 0.46, `mean_down_up_ratio` 0.45, `top_dstport_share` 0.44, `no_reply_ratio` 0.43. Scan, concentration and packet-size features dominate, in line with the Phase 1 findings.
- Lag importance, oldest to newest of the 6 windows: 8%, 11%, 15%, 17%, 19%, **29%**. The current window counts most and influence rises steadily towards it, but the full history is used.
- In-sample examples (demo model): a DDoS window gets P = 1.00 driven by `top_pair_share`, `large_pkt_ratio` and `max_ports_per_src`. A PortScan window gets P = 0.19 and a Bot window P = 0.23. For the PortScan window the drivers are small (|SHAP| about 0.6 or less, feature z-scores near zero): nothing in the state stands out. The Bot window has a few moderate drivers (`top_pair_share` +1.56 at z = +2.7) offset by negative ones. The stage head labels both as "Benign".

## Limitations

- Explanations describe what the model does, not ground truth about the attack. For attack types the model fails on (Bot, PortScan, Hulk), the explanation shows weak, diffuse evidence rather than a reason to trust the output.
- SHAP values are relative to a benign reference set; a different reference changes the values (not their sign logic).
- Stage explanations inherit the weak stage head (Phase 2): the explained stage is often "Benign" for real attacks.
- Features are correlated (for example `top_pair_share` and `top_dstport_share`), so attribution between correlated features is split somewhat arbitrarily.
- The shipped explanation table is for the demo model, which was trained on all days; it illustrates the output, it is not an evaluation. Faithfulness was checked on the holdout model on unseen data. All numbers here are for the revised training objective (see [WORLD_MODEL.md](WORLD_MODEL.md)); the original-objective values are in `models/v1_original_objective/`.

## Reproduce

```bash
python -W ignore src/explain.py      # about 4 minutes
```

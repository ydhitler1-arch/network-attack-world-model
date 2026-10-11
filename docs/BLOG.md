# What a world model can and can't learn from CIC-IDS2017

*A negative result, and why I trust it.*

I wanted to know whether a model that learns how a network's state changes from minute to minute could warn about an intrusion before a normal classifier would. I built the whole pipeline, evaluated it carefully, and the answer on this dataset is no. This post explains what I built, what I found, and what the process taught me.

## The idea

Most intrusion detection is reactive: classify a flow or a time window as benign or malicious after it happens. A **world model** takes a different view. It learns the dynamics of the network, "given the state now, what does the state look like next?", and rolls those predictions forward to estimate risk several steps ahead. If attacks have a build-up (reconnaissance, probing, then the strike), such a model could notice the build-up.

## What I built

- **State vector.** Flows from CIC-IDS2017 (2.83M flows over five days) are grouped into 60-second windows. Each window becomes 43 numbers: packet and byte rates, how many distinct hosts and ports were touched, how concentrated the traffic is on one source or one port, TCP flag shares, inter-arrival times, and scan signatures such as the most ports any single source touched. A PCAP path (Scapy) can also extract TTL, fragmentation and retransmission features, which CSV data cannot supply.
- **The model.** One LSTM cell (about 100k parameters) reads the last 6 windows and predicts a Gaussian over the next state. A 6-step rollout feeds its own predictions back in. At every step, extra heads read an infiltration probability and a MITRE ATT&CK stage.
- **Explanations.** SHAP values for every output, plus a counterfactual that removes one host's traffic and re-scores.
- **A demo.** A Streamlit app that runs offline on a CSV or PCAP.

## Result 1: there is nothing to forecast at this resolution

My first experiment, before any world model, asked a simple question: do the six minutes before an attack look different from normal? For DDoS, DoS Hulk and PortScan they do not. Attacks start abruptly. There are only 41 attack episodes in the data, and the timestamps are rounded to the minute, so any sub-minute build-up is invisible. Predicting an attack 3 minutes ahead scored a PR-AUC of 0.18 to 0.25 for every model I tried, against a base rate of 0.11, with no significant difference between models. Forecasting, at this resolution, is at chance.

## Result 2: the world model loses to logistic regression

I then asked a fairer question: forget forecasting, can the world model at least detect the current window better? I compared it with an LSTM classifier (the same network without the dynamics) and plain logistic regression, using the same features and the same evaluation:

| model | PR-AUC [95% CI] | F1 |
|---|---|---|
| Logistic regression | 0.451 [0.23, 0.67] | 0.43 |
| LSTM classifier | 0.350 [0.16, 0.59] | 0.31 |
| World model | 0.299 [0.14, 0.53] | 0.30 |

The numbers above are for the final version of the world model. The first version scored PR-AUC 0.219 and was significantly worse than logistic regression (-0.22, CI [-0.41, -0.05]); the table shows what happened after I diagnosed and fixed its training objective, described below. Even now the world model has the lowest point estimate, but the paired differences include zero. On held-out attack types the picture is weak for every model: mean recall 0.36 for logistic regression against 0.21 for the world model. Its learned dynamics tie a one-line linear AR(1) predictor at one step ahead and lose to it further out.

My first guess was overfitting: a 100k-parameter recurrent model trained on about 1,500 heavily overlapping samples. I tested it, and it is mostly wrong. Shrinking the world model to 2,500 parameters, adding dropout and weight decay, or down-weighting the dynamics loss left test PR-AUC at 0.20 to 0.25, still far below logistic regression's 0.45. Even logistic regression, with 44 parameters, scores 0.87 on its own training data and 0.45 on test, so a big train-test gap is not specific to large models; traffic shifts between blocks and days. The plain LSTM classifier did improve when made smaller (0.35 to 0.40), but the world model was worse than the classifier at every size, so the damage comes from how the world model is trained, not how big it is. My next suspect was the training objective: the world model was also trained to predict attack labels from its own imagined future states, which are close to unpredictable here. I wrote down a prediction and tested it. Removing that term raised test PR-AUC from 0.22 to 0.30 (0.20 to 0.33 for the smaller model), and training with no rollout at all reached 0.33 to 0.38, close to the plain classifier. So the objective explains much of the gap. It does not make the world model better than the simple models: the fixed version only catches up with the classifier, and logistic regression (0.45) still has the higher point estimate. I did not run a paired significance test on the improvement itself. Details: [OVERFITTING_TEST.md](OVERFITTING_TEST.md).

## Why I trust the negative result

A negative result is only worth something if the evaluation is sound. The parts that mattered:

- **Purged, blocked cross-validation.** Windows overlap in time, so a random split leaks: a test window's neighbours sit in the training set. I cut the timeline into 60-window blocks, assigned blocks to folds, and removed a gap of 12 windows around every test block from training.
- **Scalers fit inside each fold**, on benign training windows only.
- **Thresholds chosen without peeking.** Each fold calibrates its alert threshold on a separate validation fold for a 2% false-alarm target, then applies it unseen. The realised false-alarm rate then ranged from 2% to 8% on test days, which shows how much thresholds drift between days.
- **Leave-one-attack-type-out.** Every window of one attack type is removed from training, and the model is tested on it. This exposed that Bot, DoS GoldenEye and DoS Hulk are missed by every model, something the average metrics hid.
- **Confidence intervals and multiple seeds.** With 41 episodes, intervals are wide. Block bootstrap and 3 seeds keep me from over-reading small differences.
- **The dumbest baselines first.** Logistic regression and a linear AR(1) predictor are what revealed the problem.

## Explanations need validating too

SHAP plots look convincing whether or not they are right. So I checked them: my expected-gradients implementation agrees with the `shap` library (r = 0.99 on 20 windows); the attributions sum to the model's output within about 10% (so they are approximate, not exact); and a deletion test shows that resetting the top three SHAP features lowers the score by 2.1 logits while resetting three random features changes it by about zero. The explanations also show *why* the model misses some attacks: for a missed bot window the attributions are small and diffuse (no feature stands out strongly), and the stage head calls it benign, so there is little for the model to latch onto.

## Things the data could not support

- No label in CIC-IDS2017 is data exfiltration, so a model cannot learn that stage.
- DoS and DDoS are "Impact" in ATT&CK, which none of the five early-stage categories cover, so I gave them their own class rather than a wrong label.
- The PCAP-only features exist in code and tests, but the trained model cannot use them because the training data has no packet-level columns.

## Takeaways

1. A fancier model is not a better model. Always run the boring baseline.
2. Most of the work in a security ML project is evaluation design, and most of the risk is leakage.
3. Document what the data cannot tell you. It is as informative as what it can.

## What next

A dataset with finer timestamps and multi-stage attacks (CIC-IDS2018 or CTU-13); a regularised or smaller world model, or a Transformer; per-host state instead of one state for the whole network; and capturing real traffic from a lab to test the PCAP path on something other than synthetic data.

Code, results and the offline demo are in this repository; start with the [README](../README.md).

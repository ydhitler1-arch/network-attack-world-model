"""
Is the threshold drift caused by calibrating on the same validation data used for early stopping?

Hypothesis. The neural models stop training when the validation loss stops improving, so they are tuned
to the validation windows; benign validation scores are then optimistically low, and a threshold set at
their 98th percentile is too lenient on genuinely unseen windows. Logistic regression has no early stopping
and its threshold held at 2.0%.

Controlled design (same blocked-CV samples as the benchmark; val blocks are split in two halves):
  A  early stopping on ALL validation blocks, calibrate on ALL (the benchmark's method; cached scores)
  C  early stopping on half 1, calibrate on half 1   (same data used twice, half the sample: control)
  B  early stopping on half 1, calibrate on half 2   (calibration data the model never saw)
Prediction if the hypothesis is right: C and A overshoot, B lands near 2%. Both orientations (halves
swapped) and 3 seeds are averaged. Logistic regression (no early stopping) should be unaffected.

    python src/calibration_bias.py      # needs models/bench_drift_scores.pkl (src/threshold_drift.py); ~8 min
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402
import threshold_drift as TD  # noqa: E402
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from world_model import stage_ids  # noqa: E402

SEEDS = (0, 1, 2)
N_BOOT = 500


def thr_from(scores, y):
    return float(np.quantile(scores[y == 0], 1 - B.FAR_TARGET))


def collect():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    St = stage_ids(state["attack_type"], base)
    n, L, K = len(state), base["lookback"], base["horizon"]
    SP = L + 3
    torch.set_num_threads(2)
    out = []
    for k in range(B.NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -base["clip_z"], base["clip_z"]).astype(np.float32)
        te_e = B.region_starts(seg, SP, test_r)
        blk = np.arange(n) // B.BLOCK
        ids = np.unique(blk[val_r])
        halves = [val_r & np.isin(blk, ids[0::2]), val_r & np.isin(blk, ids[1::2])]
        assert not (halves[0] & halves[1]).any()
        rec = {"y_test": Y[te_e + L - 1], "seg_test": seg[te_e + L - 1], "block_test": (te_e + L - 1) // B.BLOCK,
               "orient": []}
        for es_i in (0, 1):
            es_r, cal_r = halves[es_i], halves[1 - es_i]
            o = {"y_es": Y[B.region_starts(seg, SP, es_r) + L - 1], "y_cal": Y[B.region_starts(seg, SP, cal_r) + L - 1], "scores": {}}
            es_e, cal_e = B.region_starts(seg, SP, es_r), B.region_starts(seg, SP, cal_r)
            for name, dyn in (("lstm_clf", False), ("world_model", True)):
                span = L + K if dyn else L
                trf, esf = B.region_starts(seg, span, train_r), B.region_starts(seg, span, es_r)
                es_s, cal_s, te_s = [], [], []
                for sd in SEEDS:
                    m = B.fit_neural(base, X, Y, St, trf, esf, dyn, 0, 42 + sd)
                    f = lambda st: TD.logit(B.neural_scores(m, X, st, L, dyn, [0])[0])
                    es_s.append(f(es_e)); cal_s.append(f(cal_e)); te_s.append(f(te_e))
                o["scores"][name] = (es_s, cal_s, te_s)
            # logistic regression: no early stopping, so halves only change the calibration sample
            tr = B.region_starts(seg, L, train_r)
            a, b, c = B.lr_fit_score(X[tr + L - 1], Y[tr + L - 1], [X[es_e + L - 1], X[cal_e + L - 1], X[te_e + L - 1]])
            o["scores"]["logreg"] = ([a], [b], [c])
            rec["orient"].append(o)
        out.append(rec)
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)
    return out


def evaluate(folds, drift_folds):
    res = {"target_far": B.FAR_TARGET, "models": {}}
    y_all = np.concatenate([f["y_test"] for f in folds])
    seg_all = np.concatenate([f["seg_test"] for f in folds])
    block_all = np.concatenate([f["block_test"] for f in folds])
    rng = np.random.default_rng(0)
    ub = np.unique(block_all)
    boot_idx = [np.concatenate([np.where(block_all == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(N_BOOT)]
    for model in ("logreg", "lstm_clf", "world_model"):
        n_seed = len(folds[0]["orient"][0]["scores"][model][0])
        variants = {}
        # A: the benchmark's method, from the cached scores (early stopping and calibration on all of val)
        alarms = [TD.make_alarms(TD.fixed_alarms, drift_folds, model, si) for si in range(len(drift_folds[0]["scores"][model][0]))]
        variants["A  ES on all val, calibrate on all val (benchmark)"] = alarms
        for label, which in (("C  ES on half, calibrate on the same half", "same"), ("B  ES on half, calibrate on the OTHER half", "other")):
            al = []
            for oi in (0, 1):
                for si in range(n_seed):
                    per_fold = []
                    for f in folds:
                        o = f["orient"][oi]
                        es_s, cal_s, te_s = o["scores"][model]
                        th = thr_from(es_s[si], o["y_es"]) if which == "same" else thr_from(cal_s[si], o["y_cal"])
                        per_fold.append(te_s[si] > th)
                    al.append(np.concatenate(per_fold))
            variants[label] = al
        res["models"][model] = {}
        boots = {}
        for label, alarms in variants.items():
            ms = [TD.metrics(a, y_all, seg_all) for a in alarms]
            bt = np.mean([[TD.fr(a[ix], y_all[ix]) for ix in boot_idx] for a in alarms], axis=0)
            boots[label] = bt
            day = {d: float(np.mean([m["fpr_by_day"][d] for m in ms])) for d in ms[0]["fpr_by_day"]}
            res["models"][model][label] = {
                "fpr": float(np.mean([m["fpr"] for m in ms])), "fpr_ci95": np.percentile(bt[:, 0], [2.5, 97.5]).tolist(),
                "recall": float(np.mean([m["recall"] for m in ms])), "recall_ci95": np.percentile(bt[:, 1], [2.5, 97.5]).tolist(),
                "f1": float(np.mean([m["f1"] for m in ms])), "fpr_by_day": day, "fpr_day_max": float(np.mean([m["fpr_day_max"] for m in ms]))}
        keys = list(variants)
        for other in keys[1:]:
            d = boots[other] - boots[keys[0]]
            res["models"][model][other]["paired_vs_A"] = {
                "fpr_diff": float(d[:, 0].mean()), "fpr_diff_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
                "recall_diff": float(d[:, 1].mean()), "recall_diff_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist()}
    return res


def main():
    drift = pickle.load(open("models/bench_drift_scores.pkl", "rb"))
    cache = Path("models/bench_calbias_scores.pkl")
    if "--reuse" in sys.argv and cache.exists():
        folds = pickle.load(open(cache, "rb"))
    else:
        folds = collect()
        pickle.dump(folds, open(cache, "wb"))
    res = evaluate(folds, drift)
    Path("models/calibration_bias.json").write_text(json.dumps(res, indent=1))
    for m, d in res["models"].items():
        print(f"\n== {m}")
        for lab, v in d.items():
            days = " ".join(f"{v['fpr_by_day'][k] * 100:4.1f}" for k in sorted(v["fpr_by_day"]))
            pv = v.get("paired_vs_A")
            extra = f" | vs A: FPR {pv['fpr_diff'] * 100:+.1f}pp [{pv['fpr_diff_ci95'][0] * 100:+.1f},{pv['fpr_diff_ci95'][1] * 100:+.1f}], recall {pv['recall_diff']:+.3f}" if pv else ""
            print(f"  {lab:48s} FPR {v['fpr'] * 100:4.1f}% [{v['fpr_ci95'][0] * 100:.1f},{v['fpr_ci95'][1] * 100:.1f}]  recall {v['recall']:.2f}  F1 {v['f1']:.2f} | by day {days}{extra}")


if __name__ == "__main__":
    main()

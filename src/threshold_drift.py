"""
Threshold drift: why a threshold calibrated once on a validation fold does not hold across days, and
whether adaptive calibration fixes it (src/benchmark.py found realised false-alarm rates of 2% to 8%
against a 2% target).

Same blocked-CV samples as the main benchmark (task k=0, 3 seeds for the neural models). Scores are
stored in logit space. Strategies, all evaluated on identical test samples:

  fixed      threshold = (1 - FAR) quantile of benign validation scores (the benchmark's method)
  recenter   label-free: subtract a trailing median of the previous W scores (robust to short attacks,
             but a sustained attack drags the median up and can hide itself); threshold calibrated on the
             re-centred validation stream
  feedback   analyst feedback: the threshold is nudged by online quantile tracking whenever a window is
             confirmed benign (labels arrive D windows late); starts from the fixed threshold
  oracle     per-day threshold at exactly FAR on that day's benign test scores (NOT deployable; reference)

    python src/threshold_drift.py        # ~4 min -> models/threshold_drift.json
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
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from world_model import stage_ids  # noqa: E402

SEEDS = (0, 1, 2)
FAR = B.FAR_TARGET
W = 60          # trailing window (samples) for re-centring
MIN_HIST = 20   # samples needed before the trailing median is used
D = 5           # feedback delay (samples)
ETA = 0.05      # feedback step, in units of the benign validation score std
N_BOOT = 500


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def collect():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    St = stage_ids(state["attack_type"], base)
    n, L, K = len(state), base["lookback"], base["horizon"]
    SP = L + 3
    torch.set_num_threads(2)
    folds = []
    for k in range(B.NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -base["clip_z"], base["clip_z"]).astype(np.float32)
        va_e, te_e = B.region_starts(seg, SP, val_r), B.region_starts(seg, SP, test_r)
        f = {"y_val": Y[va_e + L - 1], "y_test": Y[te_e + L - 1], "t_test": te_e + L - 1,
             "seg_test": seg[te_e + L - 1], "block_test": (te_e + L - 1) // B.BLOCK, "scores": {}}
        tr = B.region_starts(seg, L, train_r)
        sv, st = B.lr_fit_score(X[tr + L - 1], Y[tr + L - 1], [X[va_e + L - 1], X[te_e + L - 1]])
        f["scores"]["logreg"] = ([sv], [st])
        for name, dyn in (("lstm_clf", False), ("world_model", True)):
            span = L + K if dyn else L
            trf, vaf = B.region_starts(seg, span, train_r), B.region_starts(seg, span, val_r)
            vs, ts = [], []
            for sd in SEEDS:
                m = B.fit_neural(base, X, Y, St, trf, vaf, dyn, 0, 42 + sd)
                vs.append(logit(B.neural_scores(m, X, va_e, L, dyn, [0])[0]))
                ts.append(logit(B.neural_scores(m, X, te_e, L, dyn, [0])[0]))
            f["scores"][name] = (vs, ts)
        folds.append(f)
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)
    return folds


# ------------------------------------------------------------------ strategies

def trailing_median(s, init):
    out = np.empty_like(s)
    for i in range(len(s)):
        out[i] = np.median(s[max(0, i - W):i]) if i >= MIN_HIST else init
    return out


def fixed_alarms(sv, yv, st, **_):
    th = np.quantile(sv[yv == 0], 1 - FAR)
    return st > th


def recenter_alarms(sv, yv, st, **_):
    init = np.median(sv)
    v2 = sv - trailing_median(sv, init)
    th = np.quantile(v2[yv == 0], 1 - FAR)
    return (st - trailing_median(st, init)) > th


def feedback_alarms(sv, yv, st, y_test, eta=ETA, delay=D, **_):
    th = np.quantile(sv[yv == 0], 1 - FAR)
    step = eta * sv[yv == 0].std()
    alarms = np.zeros(len(st), bool)
    for i in range(len(st)):
        j = i - delay
        if j >= 0 and y_test[j] == 0:                # window j is now confirmed benign
            th += step * (float(alarms[j]) - FAR)
        alarms[i] = st[i] > th
    return alarms


def oracle_alarms(sv, yv, st, y_test, seg_test, **_):
    alarms = np.zeros(len(st), bool)
    for d in np.unique(seg_test):
        m = seg_test == d
        b = m & (y_test == 0)
        alarms[m] = st[m] > np.quantile(st[b], 1 - FAR)
    return alarms


def finite_sample_alarms(sv, yv, st, delta=0.9, **_):
    """Conservative threshold: the m-th largest benign validation score, with m the largest rank for which the
    true exceedance probability (Beta(m, n-m+1) for the m-th largest of n i.i.d. scores) is <= FAR with
    probability >= delta. delta=0.5 is roughly the usual quantile; larger delta trades recall for stability."""
    from scipy.stats import beta
    b = np.sort(sv[yv == 0])[::-1]
    n = len(b)
    m_best = 1
    for m in range(1, n + 1):
        if beta.cdf(FAR, m, n - m + 1) >= delta:
            m_best = m
        else:
            break
    return st > b[m_best - 1]


def conf80_alarms(**kw):
    return finite_sample_alarms(delta=0.8, **kw)


def conf90_alarms(**kw):
    return finite_sample_alarms(delta=0.9, **kw)


STRATEGIES = {"fixed": fixed_alarms, "recenter": recenter_alarms, "feedback": feedback_alarms, "conf80": conf80_alarms, "conf90": conf90_alarms, "oracle (reference)": oracle_alarms}


def metrics(alarm, y, seg, blocks=None):
    ben, atk = y == 0, y == 1
    fpr = float((alarm & ben).sum() / max(ben.sum(), 1))
    rec = float((alarm & atk).sum() / max(atk.sum(), 1))
    prec = float((alarm & atk).sum() / max(alarm.sum(), 1))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    day = {int(d): float((alarm & ben & (seg == d)).sum() / max((ben & (seg == d)).sum(), 1)) for d in np.unique(seg)}
    return {"fpr": fpr, "recall": rec, "precision": prec, "f1": f1, "fpr_by_day": day,
            "fpr_day_mad": float(np.mean([abs(v - FAR) for v in day.values()])),
            "fpr_day_max": float(max(day.values()))}


def fr(alarm, y):
    ben, atk = y == 0, y == 1
    return float((alarm & ben).sum() / max(ben.sum(), 1)), float((alarm & atk).sum() / max(atk.sum(), 1))


def make_alarms(fn, folds, model, si, **kw):
    out = []
    for f in folds:
        sv, st = f["scores"][model][0][si], f["scores"][model][1][si]
        out.append(fn(sv=sv, yv=f["y_val"], st=st, y_test=f["y_test"], seg_test=f["seg_test"], **kw))
    return np.concatenate(out)


def evaluate(folds):
    out = {"target_far": FAR, "params": {"W": W, "min_hist": MIN_HIST, "feedback_delay": D, "eta": ETA},
           "models": {}, "paired_feedback_minus_fixed": {}, "feedback_sensitivity": {}}
    y_all = np.concatenate([f["y_test"] for f in folds])
    seg_all = np.concatenate([f["seg_test"] for f in folds])
    block_all = np.concatenate([f["block_test"] for f in folds])
    rng = np.random.default_rng(0)
    ub = np.unique(block_all)
    boot_idx = [np.concatenate([np.where(block_all == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(N_BOOT)]
    for model in ("logreg", "lstm_clf", "world_model"):
        n_seed = len(folds[0]["scores"][model][0])
        out["models"][model] = {}
        boots = {}
        for sname, fn in STRATEGIES.items():
            alarms = [make_alarms(fn, folds, model, si) for si in range(n_seed)]
            per_seed = [metrics(a, y_all, seg_all) for a in alarms]
            b = np.mean([[fr(a[ix], y_all[ix]) for ix in boot_idx] for a in alarms], axis=0)  # (N_BOOT, 2)
            boots[sname] = b
            avg = lambda key: float(np.mean([m[key] for m in per_seed]))
            day = {d: float(np.mean([m["fpr_by_day"][d] for m in per_seed])) for d in per_seed[0]["fpr_by_day"]}
            out["models"][model][sname] = {
                "fpr": avg("fpr"), "fpr_ci95": np.percentile(b[:, 0], [2.5, 97.5]).tolist(),
                "recall": avg("recall"), "recall_ci95": np.percentile(b[:, 1], [2.5, 97.5]).tolist(),
                "precision": avg("precision"), "f1": avg("f1"),
                "fpr_by_day": day, "fpr_day_mad": avg("fpr_day_mad"), "fpr_day_max": avg("fpr_day_max")}
        # paired bootstrap, same resampled blocks: feedback minus fixed (and recenter minus fixed)
        for other in ("feedback", "recenter", "conf80", "conf90"):
            d = boots[other] - boots["fixed"]
            out["paired_feedback_minus_fixed"].setdefault(model, {})[other] = {
                "fpr_diff": float(d[:, 0].mean()), "fpr_diff_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
                "recall_diff": float(d[:, 1].mean()), "recall_diff_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist()}
        # sensitivity of the feedback strategy to its two parameters (not tuned: all reported)
        sens = {}
        for eta in (0.02, 0.05, 0.1):
            for delay in (5, 30):
                al = [make_alarms(feedback_alarms, folds, model, si, eta=eta, delay=delay) for si in range(n_seed)]
                m = [metrics(a, y_all, seg_all) for a in al]
                sens[f"eta={eta},delay={delay}"] = {k: float(np.mean([x[k] for x in m])) for k in ("fpr", "recall", "f1", "fpr_day_max")}
        out["feedback_sensitivity"][model] = sens
    out["n_test"], out["attack_rate"] = int(len(y_all)), float(y_all.mean())
    return out


def main():
    cache = Path("models/bench_drift_scores.pkl")  # git-ignored (bench_*.pkl)
    if "--reuse" in sys.argv and cache.exists():
        folds = pickle.load(open(cache, "rb"))
    else:
        folds = collect()
        pickle.dump(folds, open(cache, "wb"))
    res = evaluate(folds)
    Path("models/threshold_drift.json").write_text(json.dumps(res, indent=1))
    print("saved models/threshold_drift.json")


if __name__ == "__main__":
    main()

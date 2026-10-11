"""
Why do the neural models alarm on unusual-volume windows? (docs/DRIFT_CAUSES.md, part E; docs/VOLUME_OOD.md)

Hypothesis: those windows are out of distribution for the networks (rare in training, extreme standardised
feature values), so the scores explode. If so, interventions that bound or remove the volume signal should cut
the false alarms on the lowest- and highest-volume windows without hurting detection:

  clip3 / clip2   clip standardised inputs to +/-3 (or +/-2) instead of +/-8
  drop_vol        drop the features that track traffic volume (|Spearman| > 0.6 with log flow count among
                  benign TRAINING windows of the fold; no test data used)
  clip3+drop_vol  both

Same blocked-CV samples, protocol and thresholds as src/threshold_drift.py (threshold = 98th percentile of benign
validation scores), 3 seeds. The baseline is the cached scores of threshold_drift.py. Logistic regression is a
control: its false-alarm rate was already stable, so it should not move.

    python src/volume_ood_test.py       # needs models/bench_drift_scores.pkl; ~20 min -> models/volume_ood_test.json
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
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402
import threshold_drift as TD  # noqa: E402
from state import STATE_COLUMNS, StateScaler, transform_raw  # noqa: E402
from world_model import stage_ids  # noqa: E402

SEEDS = (0, 1, 2)
N_BOOT = 500
CONFIGS = {"clip3": (3.0, False), "clip2": (2.0, False), "drop_vol": (None, True), "clip3+drop_vol": (3.0, True)}
RHO = 0.6


def volume_features(state, mask):
    """Indices of features with |Spearman| > RHO against log flow count among the masked (benign training) windows."""
    raw = transform_raw(state[mask], STATE_COLUMNS)
    v = rankdata(np.log1p(state.loc[mask, "flow_count"].to_numpy()))
    rho = np.array([np.corrcoef(rankdata(raw[:, j]), v)[0, 1] for j in range(raw.shape[1])])
    return np.where(np.abs(rho) > RHO)[0]


def collect():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    St = stage_ids(state["attack_type"], base)
    n, L, K = len(state), base["lookback"], base["horizon"]
    SP = L + 3
    torch.set_num_threads(2)
    folds, dropped = [], {}
    for k in range(B.NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        Xfull = scaler.transform(state)
        vol = volume_features(state, train_r & (Y == 0))
        dropped[k] = [STATE_COLUMNS[j] for j in vol]
        keep = np.array([j for j in range(Xfull.shape[1]) if j not in set(vol)])
        va_e, te_e = B.region_starts(seg, SP, val_r), B.region_starts(seg, SP, test_r)
        f = {"y_val": Y[va_e + L - 1], "y_test": Y[te_e + L - 1], "t_test": te_e + L - 1, "seg_test": seg[te_e + L - 1],
             "block_test": (te_e + L - 1) // B.BLOCK, "scores": {}}
        for cname, (clip, drop) in CONFIGS.items():
            X = np.clip(Xfull, -(clip or base["clip_z"]), clip or base["clip_z"]).astype(np.float32)
            if drop:
                X = X[:, keep]
            tr = B.region_starts(seg, L, train_r)
            sv, st = B.lr_fit_score(X[tr + L - 1], Y[tr + L - 1], [X[va_e + L - 1], X[te_e + L - 1]])
            f["scores"][(cname, "logreg")] = ([sv], [st])
            for name, dyn in (("lstm_clf", False), ("world_model", True)):
                span = L + K if dyn else L
                trf, vaf = B.region_starts(seg, span, train_r), B.region_starts(seg, span, val_r)
                vs, ts = [], []
                for sd in SEEDS:
                    m = B.fit_neural(base, X, Y, St, trf, vaf, dyn, 0, 42 + sd)
                    vs.append(TD.logit(B.neural_scores(m, X, va_e, L, dyn, [0])[0]))
                    ts.append(TD.logit(B.neural_scores(m, X, te_e, L, dyn, [0])[0]))
                f["scores"][(cname, name)] = (vs, ts)
        folds.append(f)
        print(f"fold {k}: {time.time() - t0:.0f}s  dropped {dropped[k]}", flush=True)
    return folds, dropped


def evaluate(new, base_folds, state):
    fc = state["flow_count"].to_numpy()
    y_all = np.concatenate([f["y_test"] for f in new])
    seg_all = np.concatenate([f["seg_test"] for f in new])
    t_all = np.concatenate([f["t_test"] for f in new])
    block_all = np.concatenate([f["block_test"] for f in new])
    ben = y_all == 0
    q = pd.qcut(fc[t_all], 5, labels=False, duplicates="drop")        # volume quintiles over all test windows
    rng = np.random.default_rng(0)
    ub = np.unique(block_all)
    boot_idx = [np.concatenate([np.where(block_all == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(N_BOOT)]
    res = {"models": {}}
    for model in ("logreg", "lstm_clf", "world_model"):
        n_seed = len(new[0]["scores"][("clip3", model)][0])
        variants = {"baseline": [TD.make_alarms(TD.fixed_alarms, base_folds, model, si) for si in range(len(base_folds[0]["scores"][model][0]))]}
        raw_scores = {"baseline": [np.concatenate([f["scores"][model][1][si] for f in base_folds]) for si in range(len(base_folds[0]["scores"][model][0]))]}
        for cname in CONFIGS:
            al, sc = [], []
            for si in range(n_seed):
                al.append(np.concatenate([TD.fixed_alarms(sv=f["scores"][(cname, model)][0][si], yv=f["y_val"], st=f["scores"][(cname, model)][1][si]) for f in new]))
                sc.append(np.concatenate([f["scores"][(cname, model)][1][si] for f in new]))
            variants[cname], raw_scores[cname] = al, sc
        res["models"][model] = {}
        boots = {}
        for vname, alarms in variants.items():
            ms = [TD.metrics(a, y_all, seg_all) for a in alarms]
            by_q = {int(i): float(np.mean([a[ben & (q == i)].mean() for a in alarms])) for i in np.unique(q)}
            pr = float(np.mean([average_precision_score(y_all, s) for s in raw_scores[vname]]))
            b = np.mean([[TD.fr(a[ix], y_all[ix]) for ix in boot_idx] for a in alarms], axis=0)
            boots[vname] = b
            res["models"][model][vname] = {
                "fpr": float(np.mean([m["fpr"] for m in ms])), "recall": float(np.mean([m["recall"] for m in ms])), "f1": float(np.mean([m["f1"] for m in ms])),
                "pr_auc": pr, "fpr_by_day": {d: float(np.mean([m["fpr_by_day"][d] for m in ms])) for d in ms[0]["fpr_by_day"]},
                "fpr_day_max": float(np.mean([m["fpr_day_max"] for m in ms])), "fpr_by_volume_quintile": by_q}
        for vname in CONFIGS:
            d = boots[vname] - boots["baseline"]
            res["models"][model][vname]["paired_vs_baseline"] = {
                "fpr_diff": float(d[:, 0].mean()), "fpr_diff_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
                "recall_diff": float(d[:, 1].mean()), "recall_diff_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist()}
    return res


def main():
    state = pd.read_parquet("data/state_v1.parquet").reset_index(drop=True)
    base_folds = pickle.load(open("models/bench_drift_scores.pkl", "rb"))
    cache = Path("models/bench_volume_ood_scores.pkl")
    if "--reuse" in sys.argv and cache.exists():
        new, dropped = pickle.load(open(cache, "rb"))
    else:
        new, dropped = collect()
        pickle.dump((new, dropped), open(cache, "wb"))
    res = evaluate(new, base_folds, state)
    res["dropped_volume_features_by_fold"] = dropped
    Path("models/volume_ood_test.json").write_text(json.dumps(res, indent=1))
    for m, d in res["models"].items():
        print(f"\n== {m}")
        for v, x in d.items():
            q = " ".join(f"{x['fpr_by_volume_quintile'][i] * 100:4.1f}" for i in sorted(x["fpr_by_volume_quintile"]))
            pv = x.get("paired_vs_baseline")
            extra = f" | FPR {pv['fpr_diff'] * 100:+.1f}pp [{pv['fpr_diff_ci95'][0] * 100:+.1f},{pv['fpr_diff_ci95'][1] * 100:+.1f}] recall {pv['recall_diff']:+.3f}" if pv else ""
            print(f"  {v:16s} FPR {x['fpr'] * 100:4.1f}% worst day {x['fpr_day_max'] * 100:4.1f}% | by volume quintile {q} | PR-AUC {x['pr_auc']:.3f} recall {x['recall']:.2f} F1 {x['f1']:.2f}{extra}")


if __name__ == "__main__":
    main()

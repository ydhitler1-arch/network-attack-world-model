"""
Do volume-conditional thresholds improve logistic regression, or do they just move along the same ROC curve?

A threshold per traffic-volume half (split at the median flow count of benign calibration windows) raised
the baseline's F1 but also its false-alarm rate, so single operating points are not comparable. This script
sweeps the target false-alarm rate for both detectors (one global threshold vs one threshold per volume half),
under the same clean protocol as src/improve_baseline.py (calibrate on a validation half the model did not
use for selection; both orientations averaged), and compares recall at MATCHED realised false-alarm rates with
a block bootstrap.

    python src/vol_threshold_curve.py       # ~1 min -> models/vol_threshold_curve.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402
from state import STATE_COLUMNS, StateScaler  # noqa: E402

ALPHAS = np.geomspace(0.004, 0.2, 30)
TARGETS = (0.02, 0.03, 0.05)
N_BOOT = 500


def main():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    fc = state["flow_count"].to_numpy()
    n, L = len(state), base["lookback"]
    SP = L + 3
    y_all, blocks, alarms = [], [], {"global": [], "vol2": []}   # alarms[method][alpha_index] -> list over (orientation) of pooled arrays
    per = {"global": [[] for _ in ALPHAS], "vol2": [[] for _ in ALPHAS]}
    for k in range(B.NFOLD):
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -base["clip_z"], base["clip_z"]).astype(np.float32)
        blk = np.arange(n) // B.BLOCK
        ids = np.unique(blk[val_r])
        halves = [val_r & np.isin(blk, ids[0::2]), val_r & np.isin(blk, ids[1::2])]
        te_e = B.region_starts(seg, SP, test_r)
        h_e = [B.region_starts(seg, SP, h) for h in halves]
        tr = B.region_starts(seg, L, train_r)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X[tr + L - 1], Y[tr + L - 1])
        s_te = m.decision_function(X[te_e + L - 1])
        t_te = te_e + L - 1
        y_all.append(Y[t_te]); blocks.append(t_te // B.BLOCK)
        for oi in (0, 1):
            cal = 1 - oi                      # calibrate on the half that is not the selection half (LR has no selection; kept for parity)
            t_cal = h_e[cal] + L - 1
            s_cal, y_cal = m.decision_function(X[t_cal]), Y[t_cal]
            ben = y_cal == 0
            edge = np.median(fc[t_cal][ben])
            hi_te = fc[t_te] > edge
            for ai, a in enumerate(ALPHAS):
                g = s_te > np.quantile(s_cal[ben], 1 - a)
                v = np.zeros(len(s_te), bool)
                for hi in (False, True):
                    in_cal = ben & ((fc[t_cal] > edge) == hi)
                    pool = s_cal[in_cal] if in_cal.sum() >= 20 else s_cal[ben]
                    v[hi_te == hi] = s_te[hi_te == hi] > np.quantile(pool, 1 - a)
                per["global"][ai].append((k, oi, g)); per["vol2"][ai].append((k, oi, v))
    y = np.concatenate(y_all); blk_all = np.concatenate(blocks)
    A = {meth: np.array([[np.concatenate([e[2] for e in per[meth][ai] if e[1] == oi]) for oi in (0, 1)] for ai in range(len(ALPHAS))]) for meth in per}
    # A[meth] shape (n_alpha, 2 orientations, N)

    def curve(meth, idx):
        yy = y[idx]
        ben, atk = yy == 0, yy == 1
        a = A[meth][:, :, idx]
        fpr = (a & ben).sum(-1) / max(ben.sum(), 1)
        rec = (a & atk).sum(-1) / max(atk.sum(), 1)
        return fpr.mean(1), rec.mean(1)   # average over orientations -> (n_alpha,)

    def recall_at(fpr, rec, target):
        o = np.argsort(fpr)
        return float(np.interp(target, fpr[o], rec[o]))

    rng = np.random.default_rng(0)
    ub = np.unique(blk_all)
    full = np.arange(len(y))
    res = {"n_test": int(len(y)), "targets": list(TARGETS), "methods": {}}
    boot = {meth: {t: [] for t in TARGETS} for meth in A}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.where(blk_all == b)[0] for b in rng.choice(ub, len(ub))])
        for meth in A:
            f_, r_ = curve(meth, idx)
            for t in TARGETS:
                boot[meth][t].append(recall_at(f_, r_, t))
    for meth in A:
        f_, r_ = curve(meth, full)
        res["methods"][meth] = {"operating_points": {f"{a:.4f}": [float(f), float(r)] for a, f, r in zip(ALPHAS, f_, r_)},
                                "recall_at_fpr": {str(t): {"value": recall_at(f_, r_, t), "ci95": np.percentile(boot[meth][t], [2.5, 97.5]).tolist()} for t in TARGETS}}
    res["paired_vol2_minus_global"] = {}
    for t in TARGETS:
        d = np.array(boot["vol2"][t]) - np.array(boot["global"][t])
        res["paired_vol2_minus_global"][str(t)] = {"mean": float(d.mean()), "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "share_vol2_better": float((d > 0).mean())}
    Path("models/vol_threshold_curve.json").write_text(json.dumps(res, indent=1))
    print("recall at matched false-alarm rate (test, clean protocol):")
    for t in TARGETS:
        g, v = res["methods"]["global"]["recall_at_fpr"][str(t)], res["methods"]["vol2"]["recall_at_fpr"][str(t)]
        p = res["paired_vol2_minus_global"][str(t)]
        print(f"  FPR {t*100:.0f}%: global {g['value']:.3f} [{g['ci95'][0]:.2f},{g['ci95'][1]:.2f}]  vol2 {v['value']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]  paired diff {p['mean']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]  vol2 better in {p['share_vol2_better']:.0%}")


if __name__ == "__main__":
    main()

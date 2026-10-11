"""
Honest evaluation for attack forecasting: baselines + campaign-grouped CV.

Why this exists
---------------
The original time-ordered 80/20 split can leave only one or two attack
campaigns in validation, so metrics are close to meaningless. Here we:

1. Group windows into attack *episodes* (campaigns) and cross-validate by
   episode blocks, so every fold tests on campaigns the model never saw.
2. Purge training samples within (lookback + horizon) of the test block, so no
   sample straddles the train/test boundary.
3. Drop samples whose lookback already contains an attack window. Those are
   "detection" samples (the attack is already visible); keeping them lets any
   model look good by just noticing the ongoing attack.
4. Fit normalisation on training folds only (no leakage from test statistics).
5. Report lead time at a FIXED false-alarm rate, plus the fraction of
   episodes warned at all. A model that only fires at onset has lead time 0.

Models compared: a one-feature heuristic (scan score), logistic regression and
XGBoost, all on hand-built lookback summaries. A sequence model (LSTM) must
beat these to justify its complexity.

Run:  python src/evaluate.py [windows.parquet] [lookback] [horizon]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules (features.py ...) live in src/
from features import FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Episodes + samples
# --------------------------------------------------------------------------
def label_episodes(is_attack: np.ndarray, merge_gap: int = 1, seg: np.ndarray | None = None):
    """Return per-window episode id (-1 = none) and the onset window of each episode."""
    seg = np.zeros(len(is_attack), dtype=int) if seg is None else seg
    ep = np.full(len(is_attack), -1, dtype=int)
    onsets: list[int] = []
    last_attack = -10**9
    cur = -1
    for i, a in enumerate(is_attack):
        if a:
            if i - last_attack > merge_gap + 1 or seg[i] != seg[last_attack]:
                cur += 1
                onsets.append(i)
            ep[i] = cur
            last_attack = i
    return ep, onsets


def build_samples(windows: pd.DataFrame, lookback: int, horizon: int, detect: bool = False):
    """
    Sample at index i uses windows[i-lookback:i] and predicts whether an attack
    starts within [i, i+horizon). Only 'clean' samples are kept: no attack in
    the lookback, so this is forecasting rather than detection.
    """
    # log1p tames the heavy-tailed rate/count features (all are >= 0).
    feats = np.log1p(np.maximum(windows[FEATURE_COLUMNS].to_numpy(dtype=np.float64), 0))
    atk = windows["window_is_attack"].to_numpy().astype(int)
    seg = windows["segment"].to_numpy() if "segment" in windows else np.zeros(len(windows), int)
    ep, onsets = label_episodes(atk, seg=seg)

    atype = windows["attack_type"].to_numpy() if "attack_type" in windows else np.array([""] * len(windows))
    rows, y, idx, sample_ep, sample_type = [], [], [], [], []
    for i in range(lookback, len(windows) - horizon):
        if detect:
            # Sanity-check mode: label = "is the LAST window of the input an attack?" (detection,
            # not forecasting). If features cannot detect even this, they are the problem.
            if seg[i - lookback] != seg[i - 1]:
                continue
            fut = atk[i - 1 : i]
            base = i - 1
        else:
            if seg[i - lookback] != seg[i + horizon - 1]:
                continue  # sample would span an overnight gap
            if atk[i - lookback : i].any():
                continue  # attack already visible in the input -> not forecasting
            fut = atk[i : i + horizon]
            base = i
        rows.append(feats[i - lookback : i])
        y.append(int(fut.any()))
        idx.append(i)
        sample_ep.append(ep[base + int(np.argmax(fut))] if fut.any() else -1)
        sample_type.append(atype[base + int(np.argmax(fut))] if fut.any() else "")
    build_samples.last_types = np.array(sample_type)
    build_samples.onset_types = np.array([atype[o] for o in onsets])
    return np.stack(rows), np.array(y), np.array(idx), np.array(sample_ep), np.array(onsets)


def summarise(seq: np.ndarray) -> np.ndarray:
    """(n, lookback, f) -> (n, 4f): last value, mean, max, and recent-vs-early slope."""
    last = seq[:, -1, :]
    mean = seq.mean(axis=1)
    mx = seq.max(axis=1)
    half = seq.shape[1] // 2
    slope = seq[:, half:, :].mean(axis=1) - seq[:, :half, :].mean(axis=1)
    return np.concatenate([last, mean, mx, slope], axis=1)


# --------------------------------------------------------------------------
# Models (all expose fit(Xtr, ytr) / score(X))
# --------------------------------------------------------------------------
class ScanHeuristic:
    """Zero-learning baseline: risk = z-score of destination-port fan-out in the last window."""

    name = "heuristic(unique_dst_ports)"

    def fit(self, X, y):
        j = FEATURE_COLUMNS.index("unique_dst_ports")
        self.j = j
        col = X[:, j]  # 'last' block comes first in summarise()
        self.mu, self.sd = col.mean(), col.std() + 1e-9
        return self

    def score(self, X):
        return (X[:, self.j] - self.mu) / self.sd


class Logit:
    name = "logistic_regression"

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        self.m.fit((X - self.mu) / self.sd, y)
        return self

    def score(self, X):
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]


class XGB:
    name = "xgboost"

    def fit(self, X, y):
        from xgboost import XGBClassifier

        spw = (1 - y.mean()) / max(y.mean(), 1e-3)
        self.m = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
            n_jobs=2, random_state=0,
        )
        self.m.fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# Grouped CV
# --------------------------------------------------------------------------
def grouped_oof_scores(model_factory, Xs, y, idx, sample_ep, onsets, n_folds, purge):
    """Out-of-fold scores. Folds are contiguous blocks of episodes."""
    n_ep = len(onsets)
    edges = np.linspace(0, n_ep, n_folds + 1).astype(int)
    scores = np.full(len(y), np.nan)
    for k in range(n_folds):
        lo_ep, hi_ep = edges[k], edges[k + 1]
        if hi_ep <= lo_ep:
            continue
        t_lo = onsets[lo_ep - 1] if lo_ep > 0 else -1  # test block starts after previous episode
        t_hi = onsets[hi_ep] if hi_ep < n_ep else 10**9
        test = (idx > t_lo) & (idx <= t_hi - 1) if hi_ep < n_ep else (idx > t_lo)
        train = (idx < (t_lo + 1) - purge) | (idx > t_hi + purge)
        if not test.any():
            continue  # nothing left to score in this fold (e.g. other types' samples removed)
        if len(np.unique(y[train])) < 2:
            continue  # no positives in the training part of this fold -> cannot fit; left as NaN
        m = model_factory().fit(Xs[train], y[train])
        scores[test] = m.score(Xs[test])
    return scores


def blocked_oof_scores(model_factory, Xs, y, idx, n_folds, purge, block=60):
    """
    Blocked CV: cut the timeline into chunks of `block` windows and deal chunks to folds
    round-robin, so every fold sees every capture day and attack type. Training samples
    within `purge` windows of any test sample are dropped, so no input window or
    forecast horizon overlaps the test data (no leakage across the boundary).
    """
    fold = (idx // block) % n_folds
    scores = np.full(len(y), np.nan)
    for k in range(n_folds):
        test = fold == k
        if not test.any():
            continue
        s_idx = np.sort(idx[test])
        pos = np.searchsorted(s_idx, idx)
        left = np.where(pos > 0, idx - s_idx[np.clip(pos - 1, 0, len(s_idx) - 1)], 10**9)
        right = np.where(pos < len(s_idx), s_idx[np.clip(pos, 0, len(s_idx) - 1)] - idx, 10**9)
        near = np.minimum(np.abs(left), np.abs(right))
        train = (~test) & (near > purge)
        if len(np.unique(y[train])) < 2:
            continue
        m = model_factory().fit(Xs[train], y[train])
        scores[test] = m.score(Xs[test])
    return scores


def lead_time_report(scores, y, idx, onsets, horizon, target_fpr=0.02):
    """Threshold at a fixed FPR on negatives; report per-episode lead time."""
    valid = ~np.isnan(scores)
    s, yy, ii = scores[valid], y[valid], idx[valid]
    fpr, tpr, thr = roc_curve(yy, s)
    j = np.searchsorted(fpr, target_fpr, side="right") - 1
    threshold = thr[max(j, 0)]
    alarm = s >= threshold

    leads = []
    for o in onsets:
        m = (ii >= o - horizon + 1) & (ii <= o)
        if not m.any():
            continue
        fired = ii[m][alarm[m]]
        leads.append(o - fired.min() if len(fired) else None)
    caught = [l for l in leads if l is not None]
    return {
        "threshold": float(threshold),
        "achieved_fpr": float(fpr[max(j, 0)]),
        "episodes": len(leads),
        "warned_frac": len(caught) / max(len(leads), 1),
        "mean_lead": float(np.mean(caught)) if caught else float("nan"),
        "median_lead": float(np.median(caught)) if caught else float("nan"),
    }


def main(path="data/windows.parquet", lookback=12, horizon=6, n_folds=5, target_fpr=0.02, by_type=False, detect=False, cv="blocked"):
    windows = pd.read_parquet(path)
    seq, y, idx, sample_ep, onsets = build_samples(windows, lookback, horizon, detect=detect)
    if detect:
        print("*** DETECTION sanity mode: predicting whether the CURRENT window is an attack (lead-time column is meaningless here) ***")
    Xs = summarise(seq)
    print(f"{len(windows)} windows | {len(onsets)} attack episodes | {len(y)} clean samples | "
          f"positive rate {y.mean():.3f}")
    if len(onsets) < n_folds * 2:
        print("WARNING: too few episodes for stable grouped CV; results are anecdotal.")

    purge = lookback + horizon
    print(f"{n_folds}-fold {cv} CV, purge={purge} windows, "
          f"alarm threshold at {target_fpr:.0%} FPR\n")
    hdr = f"{'model':32s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'warned':>7s} {'lead(mean/med)':>15s}"
    print(hdr)
    print("-" * len(hdr))
    for factory in (ScanHeuristic, Logit, XGB):
        sc = (blocked_oof_scores(factory, Xs, y, idx, n_folds, purge)
              if cv == "blocked" else
              grouped_oof_scores(factory, Xs, y, idx, sample_ep, onsets, n_folds, purge))
        v = ~np.isnan(sc)
        ap = average_precision_score(y[v], sc[v])
        auc = roc_auc_score(y[v], sc[v])
        r = lead_time_report(sc, y, idx, onsets, horizon, target_fpr)
        print(f"{factory.name:32s} {ap:7.3f} {auc:8.3f} {r['warned_frac']:7.0%} "
              f"{r['mean_lead']:7.1f}/{r['median_lead']:<6.1f}")
    print(f"\nBase rate (random PR-AUC): {y.mean():.3f}. Lead time is in windows before onset.")

    if by_type:
        stypes, otypes = build_samples.last_types, build_samples.onset_types
        print("\n=== Per attack type (positives = that type only; other types' samples removed) ===")
        print(f"episodes per type: { {t: int((otypes == t).sum()) for t in sorted(set(otypes))} }\n")
        hdr = f"{'attack type':34s} {'eps':>4s} {'model':>20s} {'PR-AUC':>7s} {'base':>6s} {'warned':>7s} {'lead':>6s}"
        print(hdr); print("-" * len(hdr))
        for t in sorted(set(otypes)):
            if (otypes == t).sum() < 3:
                continue
            keep = (y == 0) | (stypes == t)
            for factory in (Logit, XGB):
                # Folds are blocks of THIS type's own episodes. Using global episode blocks would
                # put a type that happens on one day (e.g. all PortScans) into a single fold,
                # leaving nothing to train on.
                t_onsets = onsets[otypes == t]
                sc_full = grouped_oof_scores(factory, Xs[keep], y[keep], idx[keep], sample_ep[keep],
                                             t_onsets, min(n_folds, len(t_onsets)), purge)
                yk = y[keep]; v = ~np.isnan(sc_full)
                if yk[v].sum() == 0:
                    continue
                ap = average_precision_score(yk[v], sc_full[v])
                r = lead_time_report(sc_full, yk, idx[keep], onsets[otypes == t], horizon, target_fpr)
                print(f"{t[:34]:34s} {(otypes == t).sum():4d} {factory.name:>20s} {ap:7.3f} "
                      f"{yk[v].mean():6.3f} {r['warned_frac']:7.0%} {r['mean_lead']:6.1f}")


if __name__ == "__main__":
    flags = [x for x in sys.argv[1:] if x.startswith("--")]
    a = [sys.argv[0]] + [x for x in sys.argv[1:] if not x.startswith("--")]
    main(
        a[1] if len(a) > 1 else "data/windows.parquet",
        int(a[2]) if len(a) > 2 else 12,
        int(a[3]) if len(a) > 3 else 6,
        by_type="--by-type" in flags,
        detect="--detect" in flags,
        cv="episodes" if "--cv-episodes" in flags else "blocked",
    )

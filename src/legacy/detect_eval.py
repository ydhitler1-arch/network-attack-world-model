"""
Early-detection evaluation: how many minutes after an attack starts does the alarm fire?

Task
----
Input : the last `lookback` windows of traffic features, INCLUDING the current one.
Target: is the current window an attack window?
(This is detection, not forecasting. The forecasting experiment found no usable
advance warning at 1-minute resolution on CIC-IDS2017; see evaluate.py.)

Metric that matters here: detection delay at a fixed false-alarm rate.
  * threshold = score quantile such that ~`--fpr` of benign windows would alarm
  * per attack episode: delay = first alarm minute - onset minute (None = missed)

CV: blocked folds + purge (see evaluate.blocked_oof_scores) so no input window
overlaps test data.

Usage:
  python src/detect_eval.py                 # heuristic, logistic regression, XGBoost + SHAP
  python src/detect_eval.py --lstm          # also LSTM (needs torch)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules (features.py ...) live in src/
from evaluate import XGB, Logit, label_episodes, summarise
from features import FEATURE_COLUMNS

STATS = ["last", "mean", "max", "slope"]  # order produced by evaluate.summarise
FEAT_NAMES = [f"{s}:{f}" for s in STATS for f in FEATURE_COLUMNS]
NF = len(FEATURE_COLUMNS)


# ----------------------------------------------------------------------------- samples
def build_detection_samples(w: pd.DataFrame, lookback: int):
    feats = np.log1p(np.maximum(w[FEATURE_COLUMNS].to_numpy(dtype=np.float64), 0))
    atk = w["window_is_attack"].to_numpy().astype(int)
    seg = w["segment"].to_numpy() if "segment" in w else np.zeros(len(w), dtype=int)
    atype = w["attack_type"].to_numpy() if "attack_type" in w else np.array([""] * len(w))
    X, y, idx = [], [], []
    for i in range(lookback - 1, len(w)):
        if seg[i - lookback + 1] != seg[i]:
            continue  # would span an overnight gap
        X.append(feats[i - lookback + 1 : i + 1])
        y.append(atk[i])
        idx.append(i)
    return np.stack(X), np.array(y), np.array(idx), atk, seg, atype


# ----------------------------------------------------------------------------- models
class Anomaly:
    """Unsupervised baseline: largest |z-score| of any feature in the current window."""

    name = "anomaly(max|z|)"

    def fit(self, X, y):
        b = X[y == 0][:, :NF]  # first block of summarise() = last window
        self.mu, self.sd = b.mean(0), b.std(0) + 1e-9
        return self

    def score(self, X):
        return np.abs((X[:, :NF] - self.mu) / self.sd).max(1)


class LogitM(Logit):
    name = "logistic_regression"


class XGBExplain(XGB):
    name = "xgboost"

    def shap(self, X):
        # XGBoost's built-in TreeSHAP (identical in kind to shap.TreeExplainer); the last
        # column is the bias term, dropped here. Values are in log-odds.
        from xgboost import DMatrix

        return self.m.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]


class LSTMDetector:
    name = "lstm"

    def __init__(self, epochs: int = 20, batch: int = 64, lr: float = 1e-3):
        self.epochs, self.batch, self.lr = epochs, batch, lr

    def fit(self, X, y):
        import torch

        from model import AttackForecastLSTM

        torch.manual_seed(0)
        self.torch = torch
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.m = AttackForecastLSTM(n_features=X.shape[-1])
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        pw = min((1 - y.mean()) / max(y.mean(), 1e-3), 20.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
        n = len(Xt)
        for _ in range(self.epochs):
            self.m.train()
            perm = torch.randperm(n)
            for s in range(0, n, self.batch):
                b = perm[s : s + self.batch]
                opt.zero_grad()
                loss_fn(self.m(Xt[b]), yt[b]).backward()
                opt.step()
        return self

    def score(self, X):
        torch = self.torch
        self.m.eval()
        with torch.no_grad():
            xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return torch.sigmoid(self.m(xt)).numpy()


# ----------------------------------------------------------------------------- CV
def run_blocked(factory, X, y, idx, n_folds, purge, block=60, explain=False):
    fold = (idx // block) % n_folds
    scores = np.full(len(y), np.nan)
    shap_vals = np.full((len(y), X.shape[1]), np.nan) if explain else None
    for k in range(n_folds):
        test = fold == k
        if not test.any():
            continue
        s_idx = np.sort(idx[test])
        pos = np.searchsorted(s_idx, idx)
        left = np.where(pos > 0, idx - s_idx[np.clip(pos - 1, 0, len(s_idx) - 1)], 10**9)
        right = np.where(pos < len(s_idx), s_idx[np.clip(pos, 0, len(s_idx) - 1)] - idx, 10**9)
        train = (~test) & (np.minimum(np.abs(left), np.abs(right)) > purge)
        if len(np.unique(y[train])) < 2:
            continue
        m = factory().fit(X[train], y[train])
        scores[test] = m.score(X[test])
        if explain:
            shap_vals[test] = m.shap(X[test])
    return scores, shap_vals


# ----------------------------------------------------------------------------- delay metric
def delay_table(scores, y, idx, atk, seg, atype, fpr):
    valid = ~np.isnan(scores)
    thr = float(np.quantile(scores[valid & (y == 0)], 1 - fpr))
    full = np.full(len(atk), np.nan)
    full[idx[valid]] = scores[valid]
    ep, onsets = label_episodes(atk, seg=seg)
    rows = []
    for k, o in enumerate(onsets):
        members = np.where(ep == k)[0]
        s = full[members]
        if np.isnan(s).all():
            continue
        hit = np.where(s >= thr)[0]
        rows.append(
            {
                "onset": int(o),
                "type": atype[o],
                "length": len(members),
                "delay": float(members[hit[0]] - o) if len(hit) else np.nan,
            }
        )
    return thr, pd.DataFrame(rows)


def summarize_delays(df):
    det = df["delay"].notna()
    return {
        "episodes": len(df),
        "detected": float(det.mean()),
        "within_1min": float((df["delay"] <= 1).mean()),
        "within_3min": float((df["delay"] <= 3).mean()),
        "median_delay": float(df.loc[det, "delay"].median()) if det.any() else float("nan"),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/windows.parquet")
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--fpr", type=float, default=0.02)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    w = pd.read_parquet(args.windows)
    seq, y, idx, atk, seg, atype = build_detection_samples(w, args.lookback)
    Xs = summarise(seq)
    print(
        f"{len(w)} windows | {len(y)} samples | attack-window rate {y.mean():.3f} | "
        f"lookback={args.lookback} min | blocked {args.folds}-fold CV, purge={args.lookback} | "
        f"threshold at {args.fpr:.0%} benign false-alarm rate\n"
    )

    runs = [(Anomaly, Xs, False), (LogitM, Xs, False), (XGBExplain, Xs, True)]
    if args.lstm:
        runs.append((LSTMDetector, seq, False))

    results, all_scores, shap_xgb, tables = {}, {}, None, {}
    hdr = f"{'model':20s} {'PR-AUC':>7s} {'ROC':>6s} {'detected':>9s} {'<=1 min':>8s} {'<=3 min':>8s} {'median delay':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for factory, X, explain in runs:
        sc, sv = run_blocked(factory, X, y, idx, args.folds, args.lookback, explain=explain)
        v = ~np.isnan(sc)
        ap_, auc = average_precision_score(y[v], sc[v]), roc_auc_score(y[v], sc[v])
        thr, df = delay_table(sc, y, idx, atk, seg, atype, args.fpr)
        s = summarize_delays(df)
        name = factory.name
        results[name] = {"pr_auc": ap_, "roc_auc": auc, "threshold": thr, **s}
        all_scores[name], tables[name] = sc, df
        if explain:
            shap_xgb = sv
        print(
            f"{name:20s} {ap_:7.3f} {auc:6.3f} {s['detected']:9.0%} {s['within_1min']:8.0%} "
            f"{s['within_3min']:8.0%} {s['median_delay']:10.1f} min"
        )
    print(f"\nBase rate (random PR-AUC): {y.mean():.3f}")

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\n=== Per attack type: {best} (best PR-AUC) ===")
    df = tables[best]
    print(f"{'attack type':30s} {'eps':>4s} {'detected':>9s} {'<=1 min':>8s} {'median delay':>13s}")
    for t, g in df.groupby("type"):
        s = summarize_delays(g)
        print(f"{t[:30]:30s} {len(g):4d} {s['detected']:9.0%} {s['within_1min']:8.0%} {s['median_delay']:10.1f} min")

    # ---- artifacts for the dashboard / write-up
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    art = pd.DataFrame(
        {
            "window_idx": idx,
            "window_start": w["window_start"].to_numpy()[idx],
            "is_attack": y,
            "attack_type": atype[idx],
            **{f"score_{k}": v for k, v in all_scores.items()},
        }
    )
    if shap_xgb is not None:
        for j, f in enumerate(FEAT_NAMES):
            art[f"shap_{f}"] = shap_xgb[:, j]
    art.to_parquet(out / "detection_oof.parquet")
    (out / "detection_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out/'detection_oof.parquet'} and {out/'detection_results.json'}")


if __name__ == "__main__":
    main()

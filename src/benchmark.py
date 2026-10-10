"""
Phase 3 benchmark: world model vs LSTM classifier vs logistic regression, same 43 state features.

    python src/benchmark.py --part cv      # blocked 5-fold CV with purge        -> models/bench_cv.pkl
    python src/benchmark.py --part loto    # leave-one-attack-type-out           -> models/bench_loto.pkl
    python src/benchmark.py --part report  # metrics, bootstrap CIs              -> models/benchmark_results.json

Protocol
- Windows are cut into blocks of 60; block b belongs to fold b % 5. Fold k's blocks are the test set;
  one other fold's blocks ((k+1) % 5) are an inner validation set (early stopping + threshold
  calibration); the rest train. Samples must lie entirely inside one region and one capture day;
  windows within `purge` (12 = lookback + horizon) of a test/validation window are removed from
  training, so no sample overlaps across regions.
- Scaler is refit per fold on benign training windows only.
- Task k=0: is the current window an attack, given the last 6 windows (detection).
  Task k=3: will the window 3 minutes ahead be an attack (forecasting; the world model reads it off a
  3-step rollout, the others are trained directly on the shifted label).
- Neural models: 3 seeds, same architecture and hyperparameters (configs/world_model.json), early
  stopping on validation infiltration loss. Nothing is tuned on test.
- Leave-one-attack-type-out: same CV, but every window of type A (plus purge margin) is removed from
  train AND validation; metrics use test samples whose target window is benign (negatives) or type A
  (positives); windows of other attack types are excluded from that metric.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from wm_train import gather, run_epoch, seed_everything  # noqa: E402
from world_model import WorldModel, _norm_label, stage_ids  # noqa: E402

CFG_PATH = "configs/world_model.json"
BLOCK, NFOLD, PURGE = 60, 5, 12
FAR_TARGET = 0.02  # benign false-alarm rate used for threshold calibration
MODELS = ["lr_static", "lr_seq", "lstm_clf", "world_model"]
TASKS = {0: "detect now (k=0)", 3: "forecast 3 min ahead (k=3)"}


# ----------------------------------------------------------------- regions / samples

def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return np.convolve(mask.astype(int), np.ones(2 * r + 1, dtype=int), mode="same") > 0


def region_starts(seg: np.ndarray, span: int, region: np.ndarray) -> np.ndarray:
    n = len(seg)
    i = np.arange(n - span + 1)
    cs = np.concatenate([[0], np.cumsum(~region)])
    ok = (seg[i] == seg[i + span - 1]) & ((cs[i + span] - cs[i]) == 0)
    return i[ok]


def regions(n: int, k: int, banned: np.ndarray | None):
    blk = np.arange(n) // BLOCK
    fold = blk % NFOLD
    test_w = fold == k
    near_test = dilate(test_w, PURGE)
    val_r = (fold == (k + 1) % NFOLD) & ~near_test
    train_r = ~test_w & (fold != (k + 1) % NFOLD) & ~near_test & ~dilate(val_r, PURGE)
    if banned is not None and banned.any():
        nb = dilate(banned, PURGE)
        val_r &= ~nb
        train_r &= ~nb
    return train_r, val_r, test_w


def seq_features(Z: np.ndarray, starts: np.ndarray, L: int) -> np.ndarray:
    h = gather(Z, starts, L)
    return np.concatenate([h[:, -1], h.mean(1), h.max(1), (h[:, -1] - h[:, 0]) / (L - 1)], axis=1)


# ----------------------------------------------------------------- neural fitting

def fit_neural(cfg, X, Y, S, tr_starts, va_starts, dyn: bool, shift: int, seed: int, max_epochs=30, patience=6):
    c = dict(cfg, use_dynamics=dyn, clf_shift=shift, seed=seed)
    L, K = c["lookback"], c["horizon"]
    Kuse = K if dyn else 0
    seed_everything(seed)
    model = WorldModel(X.shape[1], len(c["stages"]), c["hidden"], c["dropout"], dyn)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    span = L + Kuse if dyn else L + shift
    tgt = tr_starts + span - 1
    pos = Y[tgt].mean()
    pos_weight = torch.tensor(min(c["pos_weight_cap"], (1 - pos) / max(pos, 1e-6)), dtype=torch.float32)
    cnt = np.bincount(S[tgt], minlength=len(c["stages"])).astype(np.float64)
    sw = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    stage_w = torch.tensor(sw / sw[cnt > 0].mean(), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, best_sd, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        run_epoch(model, X, Y, S, tr_starts, c, Kuse, pos_weight, stage_w, opt, gen)
        va = run_epoch(model, X, Y, S, va_starts, c, Kuse, pos_weight, stage_w)
        if va["bce"] < best - 1e-4:
            best, bad = va["bce"], 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_sd)
    return model.eval()


@torch.no_grad()
def neural_scores(model, X, starts, L, dyn, tasks):
    x = torch.from_numpy(gather(X, starts, L))
    o = model(x, max(tasks) if dyn else 0)
    p = torch.sigmoid(o["atk"]).numpy()
    return {k: p[:, L - 1 + (k if dyn else 0)] for k in tasks}


def lr_fit_score(Ztr_f, ytr, Zeval_fs):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(Ztr_f, ytr)
    return [clf.decision_function(f) for f in Zeval_fs]


def calibrate(scores_val, y_val):
    return float(np.quantile(scores_val[y_val == 0], 1 - FAR_TARGET))


# ----------------------------------------------------------------- CV driver

def run_cv(cfg, state, banned=None, tasks=(0, 3), seeds=(0, 1, 2), models=MODELS, log=print):
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    St = stage_ids(state["attack_type"], cfg)
    atype = state["attack_type"].map(_norm_label).to_numpy()
    n, L, K = len(state), cfg["lookback"], cfg["horizon"]
    SP = L + max(tasks)  # evaluation span: history + furthest target
    sc_models = [m for m in models]
    out = {m: {k: {"score": [], "thr": []} for k in tasks} for m in sc_models}
    meta = {"y": {k: [] for k in tasks}, "block": [], "atype": [], "fold": []}

    for k in range(NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = regions(n, k, banned)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -cfg["clip_z"], cfg["clip_z"]).astype(np.float32)
        va_e = region_starts(seg, SP, val_r)
        te_e = region_starts(seg, SP, test_r)
        tt = te_e + L - 1
        meta["block"].append(tt // BLOCK)
        meta["atype"].append(atype[tt])
        meta["fold"].append(np.full(len(tt), k))
        for t in tasks:
            meta["y"][t].append(Y[tt + t])
        yv = {t: Y[va_e + L - 1 + t] for t in tasks}

        def record(m, t, s_val, s_te):
            out[m][t]["score"].append(s_te)
            out[m][t]["thr"].append(np.full(len(s_te), calibrate(s_val, yv[t])))

        # logistic regression (deterministic, seed-free): store S=1 seed dimension as list
        for m in [x for x in ("lr_static", "lr_seq") if x in sc_models]:
            for t in tasks:
                tr = region_starts(seg, L + t, train_r)
                feat = (lambda st: X[st + L - 1]) if m == "lr_static" else (lambda st: seq_features(X, st, L))
                s_val, s_te = lr_fit_score(feat(tr), Y[tr + L - 1 + t], [feat(va_e), feat(te_e)])
                record(m, t, s_val, s_te)
        # neural models
        for m, dyn in (("lstm_clf", False), ("world_model", True)):
            if m not in sc_models:
                continue
            per_seed = {t: {"s": [], "v": []} for t in tasks}
            for sd in seeds:
                if dyn:  # one fit serves all tasks (rollout)
                    tr = region_starts(seg, L + K, train_r)
                    va = region_starts(seg, L + K, val_r)
                    mod = fit_neural(cfg, X, Y, St, tr, va, True, 0, 42 + sd)
                    sv, ste = neural_scores(mod, X, va_e, L, True, tasks), neural_scores(mod, X, te_e, L, True, tasks)
                    for t in tasks:
                        per_seed[t]["s"].append(ste[t]); per_seed[t]["v"].append(sv[t])
                else:
                    for t in tasks:
                        tr = region_starts(seg, L + t, train_r)
                        va = region_starts(seg, L + t, val_r)
                        mod = fit_neural(cfg, X, Y, St, tr, va, False, t, 42 + sd)
                        per_seed[t]["s"].append(neural_scores(mod, X, te_e, L, False, [0])[0])
                        per_seed[t]["v"].append(neural_scores(mod, X, va_e, L, False, [0])[0])
            for t in tasks:
                out[m][t]["score"].append(np.stack(per_seed[t]["s"]))           # (S, N)
                out[m][t]["thr"].append(np.stack([np.full(len(te_e), calibrate(v, yv[t])) for v in per_seed[t]["v"]]))
        log(f"  fold {k}: {len(te_e)} test / {len(va_e)} val samples, {time.time() - t0:.0f}s")

    def cat(lst):
        # LR entries are (N,), neural are (S,N): normalise to (S,N)
        lst = [a if a.ndim == 2 else a[None] for a in lst]
        return np.concatenate(lst, axis=1)

    res = {m: {t: {"score": cat(out[m][t]["score"]), "thr": cat(out[m][t]["thr"])} for t in tasks} for m in sc_models}
    meta = {"y": {t: np.concatenate(v) for t, v in meta["y"].items()},
            **{k: np.concatenate(meta[k]) for k in ("block", "atype", "fold")}}
    return {"res": res, "meta": meta}


# ----------------------------------------------------------------- metrics

def point_metrics(y, s, thr):
    """y (N,), s (N,), thr (N,) per-sample calibrated threshold. Returns dict."""
    d = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    pred = s > thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    d.update(precision=pr, recall=rc, f1=(2 * pr * rc / (pr + rc) if pr + rc else 0.0),
             fpr=fp / (fp + tn) if fp + tn else 0.0)
    # oracle operating point: threshold at exactly FAR_TARGET on pooled benign scores (optimistic)
    th = np.quantile(s[y == 0], 1 - FAR_TARGET)
    po = s > th
    d["recall_at_2pct_fpr_oracle"] = float((po & (y == 1)).sum() / max((y == 1).sum(), 1))
    return d


def seed_avg(y, S, T, rows):
    ms = [point_metrics(y[rows], S[i][rows], T[i][rows]) for i in range(S.shape[0])]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, {k: float(np.std([m[k] for m in ms])) for k in ms[0]}


def block_bootstrap(y, S, T, blocks, rows, fn, B=300, seed=0):
    """fn(point-metric dict) -> scalar; resample blocks with replacement; seed-averaged."""
    rng = np.random.default_rng(seed)
    rows_idx = np.where(rows)[0]
    ub = np.unique(blocks[rows_idx])
    by_block = {b: rows_idx[blocks[rows_idx] == b] for b in ub}
    vals = []
    for _ in range(B):
        pick = rng.choice(ub, len(ub))
        idx = np.concatenate([by_block[b] for b in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(np.mean([fn(point_metrics(y[idx], S[i][idx], T[i][idx])) for i in range(S.shape[0])]))
    return float(np.mean(vals)), np.percentile(vals, [2.5, 97.5]).tolist(), np.array(vals)


def report(cv, loto):
    out = {"cv": {}, "loto": {}, "paired": {}}
    meta = cv["meta"]
    for t in TASKS:
        if t not in cv["res"]["lr_seq"]:
            continue
        y = meta["y"][t]
        rows = np.ones(len(y), bool)
        tab = {}
        for m in MODELS:
            S, T = cv["res"][m][t]["score"], cv["res"][m][t]["thr"]
            mean, sd = seed_avg(y, S, T, rows)
            ci_pr = block_bootstrap(y, S, T, meta["block"], rows, lambda d: d["pr_auc"])[:2]
            ci_f1 = block_bootstrap(y, S, T, meta["block"], rows, lambda d: d["f1"])[:2]
            tab[m] = {"mean": mean, "sd_over_seeds": sd, "pr_auc_ci95": ci_pr[1], "f1_ci95": ci_f1[1]}
        out["cv"][t] = {"n": int(len(y)), "attack_rate": float(y.mean()), "models": tab}
        # paired differences world_model - baseline (block bootstrap, same resamples)
        for base in ("lr_static", "lr_seq", "lstm_clf"):
            for met in ("pr_auc", "f1"):
                a = block_bootstrap(y, cv["res"]["world_model"][t]["score"], cv["res"]["world_model"][t]["thr"],
                                    meta["block"], rows, lambda d, m=met: d[m], seed=1)[2]
                b = block_bootstrap(y, cv["res"][base][t]["score"], cv["res"][base][t]["thr"],
                                    meta["block"], rows, lambda d, m=met: d[m], seed=1)[2]
                k = min(len(a), len(b))
                diff = a[:k] - b[:k]
                out["paired"][f"k{t}:world_model-{base}:{met}"] = {
                    "mean": float(diff.mean()), "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
    if loto:
        for A, d in loto.items():
            meta_l = d["meta"]
            sel = (meta_l["atype"] == A) | (meta_l["y"][0] == 0)
            y = (meta_l["atype"] == A).astype(int)
            row = {"n_pos": int((y[sel] == 1).sum()), "n_neg": int((y[sel] == 0).sum())}
            for m in MODELS:
                S, T = d["res"][m][0]["score"], d["res"][m][0]["thr"]
                if row["n_pos"] and row["n_neg"]:
                    mean, sd = seed_avg(y, S, T, sel)
                    row[m] = {**mean, "sd_over_seeds": sd["pr_auc"]}
            out["loto"][A] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cv", "loto", "report"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--types", nargs="*", help="LOTO: restrict to these attack types")
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(Path(CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(cfg["state_path"]).reset_index(drop=True)

    if args.part == "cv":
        t0 = time.time()
        cv = run_cv(cfg, state, seeds=tuple(range(args.seeds)))
        pickle.dump(cv, open("models/bench_cv.pkl", "wb"))
        print(f"cv done in {time.time() - t0:.0f}s")
    elif args.part == "loto":
        types = args.types or sorted({_norm_label(a) for a in state["attack_type"].unique() if a})
        loto = {}
        for A in types:
            t0 = time.time()
            banned = (state["attack_type"].map(_norm_label) == A).to_numpy()
            print(f"LOTO {A}: {int(banned.sum())} attack windows held out")
            loto[A] = run_cv(cfg, state, banned=banned, tasks=(0,), seeds=(0,), log=lambda *_: None)
            print(f"   done in {time.time() - t0:.0f}s")
            pickle.dump(loto, open("models/bench_loto.pkl", "wb"))
    else:
        cv = pickle.load(open("models/bench_cv.pkl", "rb"))
        lp = Path("models/bench_loto.pkl")
        loto = pickle.load(open(lp, "rb")) if lp.exists() else {}
        res = report(cv, loto)
        Path("models/benchmark_results.json").write_text(json.dumps(res, indent=1))
        print("saved models/benchmark_results.json")


if __name__ == "__main__":
    main()

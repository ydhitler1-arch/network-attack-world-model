"""
Does overfitting explain why the world model loses to logistic regression?

Hypothesis: a ~100k-parameter recurrent model overfits ~1,500 heavily overlapping training samples.
Predictions if true:
  (P1) train PR-AUC >> test PR-AUC for the neural models (a large generalisation gap);
  (P2) shrinking capacity (hidden units), adding regularisation, or down-weighting the dynamics loss
       narrows the gap and raises TEST PR-AUC towards logistic regression;
  (P3) validation PR-AUC predicts which config is best, so it can be chosen without peeking at test.
If instead small models stay well below logistic regression, capacity/overfitting is not the (whole) cause.

Same protocol and samples as src/benchmark.py (blocked 5-fold CV, purge 12, scaler per fold, task k=0,
3 seeds). No selection on test: the config shown as "selected" is the one with the best VALIDATION PR-AUC.

    python src/overfit_test.py            # ~10 min  -> models/overfit_test.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402
from state import STATE_COLUMNS, StateScaler  # noqa: E402
from wm_train import run_epoch, seed_everything  # noqa: E402
from world_model import WorldModel, stage_ids  # noqa: E402

SEEDS = (0, 1, 2)
# (name, model kind, config overrides)
CONFIGS = [
    ("wm_h128 (shipped)", "wm", {}),
    ("wm_h64", "wm", {"hidden": 64}),
    ("wm_h32", "wm", {"hidden": 32}),
    ("wm_h16", "wm", {"hidden": 16}),
    ("wm_h8", "wm", {"hidden": 8}),
    ("wm_h32_reg (drop .3, wd .05)", "wm", {"hidden": 32, "dropout": 0.3, "weight_decay": 0.05}),
    ("wm_h128_dyn0.1", "wm", {"loss_weights": {"dynamics": 0.1, "infiltration": 1.0, "stage": 0.5}}),
    ("wm_h32_dyn0.1", "wm", {"hidden": 32, "loss_weights": {"dynamics": 0.1, "infiltration": 1.0, "stage": 0.5}}),
    ("clf_h128", "clf", {}),
    ("clf_h32", "clf", {"hidden": 32}),
    ("clf_h16", "clf", {"hidden": 16}),
    ("clf_h8", "clf", {"hidden": 8}),
]


def fit_tracked(cfg, X, Y, S, tr, va, dyn, seed, max_epochs=30, patience=6):
    """Like benchmark.fit_neural, but also returns the best epoch and the final train/val BCE."""
    c = dict(cfg, use_dynamics=dyn, clf_shift=0, seed=seed)
    L, K = c["lookback"], c["horizon"]
    Kuse = c.get("train_horizon", K) if dyn else 0
    seed_everything(seed)
    model = WorldModel(X.shape[1], len(c["stages"]), c["hidden"], c["dropout"], dyn)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    tgt = tr + (L + Kuse) - 1
    pos = Y[tgt].mean()
    pos_weight = torch.tensor(min(c["pos_weight_cap"], (1 - pos) / max(pos, 1e-6)), dtype=torch.float32)
    cnt = np.bincount(S[tgt], minlength=len(c["stages"])).astype(np.float64)
    sw = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    stage_w = torch.tensor(sw / sw[cnt > 0].mean(), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, best_sd, best_ep, bad, info = float("inf"), None, 0, 0, {}
    for ep in range(1, max_epochs + 1):
        trl = run_epoch(model, X, Y, S, tr, c, Kuse, pos_weight, stage_w, opt, gen)
        val = run_epoch(model, X, Y, S, va, c, Kuse, pos_weight, stage_w)
        if val["bce"] < best - 1e-4:
            best, bad, best_ep = val["bce"], 0, ep
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            info = {"train_bce": trl["bce"], "val_bce": val["bce"]}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_sd)
    return model.eval(), best_ep, info, sum(p.numel() for p in model.parameters())


# Second experiment: is the training OBJECTIVE the cause? (see docs/OVERFITTING_TEST.md, follow-up)
# Prediction made before running: if training the infiltration head on imagined future states is what
# hurts detection, "realstep" should recover the classifier's level (test PR-AUC >= ~0.35 at the same
# size) and beat the shipped world model significantly; if it stays near 0.2, that hypothesis is wrong.
RS = {"rollout_task_loss": False}
OBJECTIVE_CONFIGS = [
    ("wm_h128 (shipped)", "wm", {}),
    ("wm_h128_realstep", "wm", RS),
    ("wm_h128_norollout", "wm", {"train_horizon": 0}),
    ("wm_h32", "wm", {"hidden": 32}),
    ("wm_h32_realstep", "wm", dict(RS, hidden=32)),
    ("wm_h32_norollout", "wm", {"hidden": 32, "train_horizon": 0}),
    ("clf_h128", "clf", {}),
    ("clf_h32", "clf", {"hidden": 32}),
]


def main(configs=None, out_path="models/overfit_test.json"):
    CONFIGS_ = configs or CONFIGS
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    St = stage_ids(state["attack_type"], base)
    n, L, K = len(state), base["lookback"], base["horizon"]
    SP = L + 3  # same evaluation samples as the main benchmark (tasks 0 and 3)
    torch.set_num_threads(2)
    names = ["logreg"] + [c[0] for c in CONFIGS_]
    store = {nm: {"train": [], "val": [], "test": [], "thr": [], "epoch": [], "params": None, "gap_bce": []} for nm in names}
    meta_y, meta_block = [], []
    t_all = time.time()
    for k in range(B.NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -base["clip_z"], base["clip_z"]).astype(np.float32)
        tr_e = B.region_starts(seg, SP, train_r)
        va_e = B.region_starts(seg, SP, val_r)
        te_e = B.region_starts(seg, SP, test_r)
        rng = np.random.default_rng(k)
        tr_eval = rng.choice(tr_e, min(len(tr_e), 1200), replace=False)
        y_of = lambda st: Y[st + L - 1]
        meta_y.append(y_of(te_e)); meta_block.append((te_e + L - 1) // B.BLOCK)

        # logistic regression (current window), same samples
        tr_fit = B.region_starts(seg, L, train_r)
        s = B.lr_fit_score(X[tr_fit + L - 1], Y[tr_fit + L - 1], [X[tr_eval + L - 1], X[va_e + L - 1], X[te_e + L - 1]])
        store["logreg"]["train"].append([s[0]]); store["logreg"]["val"].append([s[1]]); store["logreg"]["test"].append([s[2]])
        store["logreg"]["thr"].append([np.full(len(te_e), B.calibrate(s[1], y_of(va_e)))])
        store["logreg"]["epoch"].append([0]); store["logreg"]["gap_bce"].append([0])

        for nm, kind, ov in CONFIGS_:
            cfg = dict(base, **ov)
            dyn = kind == "wm"
            span = L + cfg.get("train_horizon", K) if dyn else L
            trf = B.region_starts(seg, span, train_r)
            vaf = B.region_starts(seg, span, val_r)
            sc_tr, sc_va, sc_te, thr, eps, gaps = [], [], [], [], [], []
            for sd in SEEDS:
                model, ep, info, nparams = fit_tracked(cfg, X, Y, St, trf, vaf, dyn, 42 + sd)
                store[nm]["params"] = nparams
                f = lambda st: B.neural_scores(model, X, st, L, dyn, [0])[0]
                sc_tr.append(f(tr_eval)); sc_va.append(f(va_e)); sc_te.append(f(te_e))
                thr.append(np.full(len(te_e), B.calibrate(sc_va[-1], y_of(va_e))))
                eps.append(ep); gaps.append(info["val_bce"] - info["train_bce"])
            store[nm]["train"].append(sc_tr); store[nm]["val"].append(sc_va); store[nm]["test"].append(sc_te)
            store[nm]["thr"].append(thr); store[nm]["epoch"].append(eps); store[nm]["gap_bce"].append(gaps)
        store["_y"] = {"train": [y_of(tr_eval)], "val": [y_of(va_e)]} if k == 0 else store.get("_y")
        store.setdefault("_ytr", []).append(y_of(tr_eval)); store.setdefault("_yva", []).append(y_of(va_e))
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)

    ytr, yva, yte = np.concatenate(store["_ytr"]), np.concatenate(store["_yva"]), np.concatenate(meta_y)
    blocks = np.concatenate(meta_block)
    out = {"n_test": int(len(yte)), "attack_rate": float(yte.mean()), "configs": {}}
    cat = lambda lst: np.concatenate([np.stack(x) for x in lst], axis=1)      # (S, N) over folds
    lr_te, lr_thr = cat(store["logreg"]["test"]), cat(store["logreg"]["thr"])
    for nm in names:
        d = store[nm]
        tr, va, te, th = cat(d["train"]), cat(d["val"]), cat(d["test"]), cat(d["thr"])
        pr = lambda y, S: [float(average_precision_score(y, s)) for s in S]
        p_tr, p_va, p_te = pr(ytr, tr), pr(yva, va), pr(yte, te)
        mean, _ = B.seed_avg(yte, te, th, np.ones(len(yte), bool))
        ci = B.block_bootstrap(yte, te, th, blocks, np.ones(len(yte), bool), lambda m: m["pr_auc"])
        dv = None
        if nm != "logreg":
            a = B.block_bootstrap(yte, te, th, blocks, np.ones(len(yte), bool), lambda m: m["pr_auc"], seed=1)[2]
            b = B.block_bootstrap(yte, lr_te, lr_thr, blocks, np.ones(len(yte), bool), lambda m: m["pr_auc"], seed=1)[2]
            m_ = min(len(a), len(b)); diff = a[:m_] - b[:m_]
            dv = {"mean": float(diff.mean()), "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
        out["configs"][nm] = {
            "params": d["params"], "pr_auc_train": [float(np.mean(p_tr)), float(np.std(p_tr))],
            "pr_auc_val": [float(np.mean(p_va)), float(np.std(p_va))], "pr_auc_test": [float(np.mean(p_te)), float(np.std(p_te))],
            "pr_auc_test_ci95": ci[1], "f1_test": mean["f1"], "fpr_test": mean["fpr"], "recall_test": mean["recall"],
            "best_epoch_mean": float(np.mean([e for fold in d["epoch"] for e in fold])),
            "val_minus_train_bce": float(np.mean([g for fold in d["gap_bce"] for g in fold])),
            "test_minus_lr_pr_auc": dv,
        }
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t_all:.0f}s -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "objective":
        main(OBJECTIVE_CONFIGS, "models/objective_test.json")
    else:
        main()

"""
Can the strongest model (logistic regression on the current window) be improved?

Same blocked-CV samples, features and test folds as src/benchmark.py. Because src/calibration_bias.py showed
that calibrating a threshold on the data used for model selection biases it, EVERY candidate here, including the
baseline, follows the same clean protocol:

    split the validation blocks into two halves
    select hyper-parameters / early-stop on half 1 (the "selection half")
    calibrate the alert threshold (2% false alarms on benign windows) on half 2 (the "calibration half")
    evaluate on the test fold; both orientations (halves swapped) are averaged

Nothing is selected on test. Candidates are all reported, with paired comparisons against the baseline; with
about a dozen candidates, one "significant" result is weak evidence on its own (see docs/IMPROVING_THE_BASELINE.md).

    python src/improve_baseline.py     # needs models/bench_calbias_scores.pkl (src/calibration_bias.py); ~5 min
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402
from state import STATE_COLUMNS, StateScaler  # noqa: E402

warnings.filterwarnings("ignore")
BASE = "lr_base"


def zscore_with(ref_scores, ref_y):
    b = ref_scores[ref_y == 0]
    mu, sd = b.mean(), b.std() + 1e-9
    return lambda s: (s - mu) / sd


def candidates_grid(Xc_tr, Xs_tr, y_tr):
    """Yield (family, config name, fitted-scorer(cur, seq) -> scores)."""
    pos = max(y_tr.mean(), 1e-6)
    spw = (1 - pos) / pos
    # logistic regression grid (current window)
    for C in (0.01, 0.03, 0.1, 0.3, 1, 3):
        for cw in ("balanced", None):
            m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(Xc_tr, y_tr)
            yield "lr_tuned", f"C={C},cw={cw}", (lambda cur, seq, m=m: m.decision_function(cur))
    # L1
    for C in (0.03, 0.1, 0.3, 1):
        m = LogisticRegression(C=C, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000).fit(Xc_tr, y_tr)
        yield "lr_l1", f"C={C}", (lambda cur, seq, m=m: m.decision_function(cur))
    # pairwise interactions among the 12 features with the largest univariate effect on the training data
    eff = np.abs(Xc_tr[y_tr == 1].mean(0) - Xc_tr[y_tr == 0].mean(0)) / (Xc_tr.std(0) + 1e-9)
    top = np.argsort(-eff)[:12]
    pf = PolynomialFeatures(2, include_bias=False)
    P_tr = pf.fit_transform(Xc_tr[:, top])
    sc = StandardScaler().fit(P_tr)
    for C in (0.01, 0.03, 0.1, 0.3):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(P_tr), y_tr)
        yield "lr_poly2", f"C={C}", (lambda cur, seq, m=m: m.decision_function(sc.transform(pf.transform(cur[:, top]))))
    # gradient-boosted trees, current window and 6-window summary
    for fam, which in (("xgb_cur", 0), ("xgb_seq", 1)):
        Xtr = Xc_tr if which == 0 else Xs_tr
        for depth in (2, 4):
            for n_est in (100, 300):
                m = XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=spw, random_state=0, n_jobs=2, eval_metric="logloss", verbosity=0).fit(Xtr, y_tr)
                yield fam, f"depth={depth},n={n_est}", (lambda cur, seq, m=m, w=which: m.predict_proba(cur if w == 0 else seq)[:, 1])
    m = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0, n_jobs=2).fit(Xc_tr, y_tr)
    yield "extra_trees", "n=300,leaf=3", (lambda cur, seq, m=m: m.predict_proba(cur)[:, 1])


def main():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    seg = state["segment"].to_numpy()
    Y = state["window_is_attack"].to_numpy(np.float32)
    n, L = len(state), base["lookback"]
    SP = L + 3
    fc = state["flow_count"].to_numpy()
    calbias = pickle.load(open("models/bench_calbias_scores.pkl", "rb"))   # neural scores, same halves/orientations
    store = {}      # name -> {"test": [(R,N) per fold], "thr": [...], "es_pr": [...]}
    meta = {"y": [], "block": []}
    t_all = time.time()
    for k in range(B.NFOLD):
        t0 = time.time()
        train_r, val_r, test_r = B.regions(n, k, None)
        scaler = StateScaler(STATE_COLUMNS).fit(state, train_r & (Y == 0))
        X = np.clip(scaler.transform(state), -base["clip_z"], base["clip_z"]).astype(np.float32)
        blk = np.arange(n) // B.BLOCK
        ids = np.unique(blk[val_r])
        halves = [val_r & np.isin(blk, ids[0::2]), val_r & np.isin(blk, ids[1::2])]
        te_e = B.region_starts(seg, SP, test_r)
        h_e = [B.region_starts(seg, SP, h) for h in halves]
        tr = B.region_starts(seg, L, train_r)
        cur = lambda st: X[st + L - 1]
        seq = lambda st: B.seq_features(X, st, L)
        y_tr, y_te = Y[tr + L - 1], Y[te_e + L - 1]
        y_h = [Y[e + L - 1] for e in h_e]
        meta["y"].append(y_te); meta["block"].append((te_e + L - 1) // B.BLOCK)

        # baseline = the benchmark's logistic regression (C=1, balanced), same clean protocol
        scored = {}   # (family, config) -> (scores on half0, half1, test)
        base_m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(cur(tr), y_tr)
        scored[(BASE, "C=1,cw=balanced")] = tuple(base_m.decision_function(cur(e)) for e in (h_e[0], h_e[1], te_e))
        for fam, cfg, fn in candidates_grid(cur(tr), seq(tr), y_tr):
            scored[(fam, cfg)] = tuple(fn(cur(e), seq(e)) for e in (h_e[0], h_e[1], te_e))
        fams = sorted({f for f, _ in scored}, key=lambda f: (f != BASE, f))
        for oi in (0, 1):          # orientation: selection half = oi, calibration half = 1 - oi
            es, cal = oi, 1 - oi
            picked = {}
            for fam in fams:
                cfgs = [c for f, c in scored if f == fam]
                prs = [average_precision_score(y_h[es], scored[(fam, c)][es]) for c in cfgs]
                bi = int(np.argmax(prs))
                picked[fam] = (cfgs[bi], prs[bi])
            for fam in fams:
                cfg, pr = picked[fam]
                s_cal, s_te = scored[(fam, cfg)][cal], scored[(fam, cfg)][2]
                thr = B.calibrate(s_cal, y_h[cal])
                d = store.setdefault(fam, {"test": [], "thr": [], "es_pr": [], "cfg": []})
                d["test"].append((k, oi, s_te)); d["thr"].append((k, oi, np.full(len(s_te), thr))); d["es_pr"].append(pr); d["cfg"].append(cfg)
            # volume-conditional thresholds ("vol2"): the same scores, but one threshold per traffic-volume half,
            # split at the median flow count of the benign calibration windows
            t_cal, t_te = h_e[cal] + L - 1, te_e + L - 1
            for fam in (BASE, "lr_tuned"):
                cfg, pr = picked[fam]
                s_cal, s_te = scored[(fam, cfg)][cal], scored[(fam, cfg)][2]
                ben = y_h[cal] == 0
                edge = np.median(fc[t_cal][ben])
                thr = np.empty(len(s_te))
                for hi in (False, True):
                    in_cal = ben & ((fc[t_cal] > edge) == hi)
                    pool = s_cal[in_cal] if in_cal.sum() >= 20 else s_cal[ben]
                    thr[(fc[t_te] > edge) == hi] = np.quantile(pool, 1 - B.FAR_TARGET)
                d = store.setdefault(f"{fam}+vol2", {"test": [], "thr": [], "es_pr": [], "cfg": []})
                d["test"].append((k, oi, s_te)); d["thr"].append((k, oi, thr)); d["es_pr"].append(pr); d["cfg"].append(cfg)
            # ensembles: z-score each member on the calibration half's benign windows, average, then threshold
            nn = {m: calbias[k]["orient"][oi]["scores"][m] for m in ("lstm_clf", "world_model")}   # (es, cal, test) lists per seed
            seeds = len(nn["lstm_clf"][0])
            members = {}
            for fam in ("lr_tuned", "xgb_cur"):
                cfg = picked[fam][0]
                members[fam] = tuple(scored[(fam, cfg)][i] for i in (es, cal, 2))
            members[BASE] = tuple(scored[(BASE, "C=1,cw=balanced")][i] for i in (es, cal, 2))
            for ename, parts in (("ens_lr+clf", [("lr_tuned", None), ("lstm_clf", "nn")]),
                                 ("ens_lr+wm", [("lr_tuned", None), ("world_model", "nn")]),
                                 ("ens_lr+clf+wm", [("lr_tuned", None), ("lstm_clf", "nn"), ("world_model", "nn")]),
                                 ("ens_lr+xgb", [("lr_tuned", None), ("xgb_cur", None)])):
                te_list, thr_list = [], []
                for si in range(seeds if any(p[1] for p in parts) else 1):
                    comp_cal, comp_te = [], []
                    for nm, kind in parts:
                        trip = (nn[nm][0][si], nn[nm][1][si], nn[nm][2][si]) if kind else members[nm]
                        z = zscore_with(trip[1], y_h[cal])
                        comp_cal.append(z(trip[1])); comp_te.append(z(trip[2]))
                    s_cal, s_te = np.mean(comp_cal, 0), np.mean(comp_te, 0)
                    te_list.append(s_te); thr_list.append(np.full(len(s_te), B.calibrate(s_cal, y_h[cal])))
                d = store.setdefault(ename, {"test": [], "thr": [], "es_pr": [], "cfg": []})
                for s_te, th in zip(te_list, thr_list):
                    d["test"].append((k, oi, s_te)); d["thr"].append((k, oi, th))
            # neural reference models under the same protocol
            for nm in ("lstm_clf", "world_model"):
                d = store.setdefault(nm, {"test": [], "thr": [], "es_pr": [], "cfg": []})
                for si in range(len(nn[nm][0])):
                    d["test"].append((k, oi, nn[nm][2][si])); d["thr"].append((k, oi, np.full(len(nn[nm][2][si]), B.calibrate(nn[nm][1][si], y_h[cal]))))
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)

    # assemble (replicates, N): replicate r = (orientation, seed) pairs, concatenated over folds
    y = np.concatenate(meta["y"]); blocks = np.concatenate(meta["block"])
    rows = np.ones(len(y), bool)

    def assemble(entries, thr_entries):
        nrep = len(entries) // B.NFOLD
        S = [np.concatenate([entries[f * nrep + r][2] for f in range(B.NFOLD)]) for r in range(nrep)]
        T = [np.concatenate([thr_entries[f * nrep + r][2] for f in range(B.NFOLD)]) for r in range(nrep)]
        return np.stack(S), np.stack(T)

    out = {"n_test": int(len(y)), "attack_rate": float(y.mean()), "candidates": {}}
    boots = {}
    for name, d in store.items():
        S, T = assemble(d["test"], d["thr"])
        mean, _ = B.seed_avg(y, S, T, rows)
        ci_pr = B.block_bootstrap(y, S, T, blocks, rows, lambda m: m["pr_auc"], B=500, seed=1)
        ci_f1 = B.block_bootstrap(y, S, T, blocks, rows, lambda m: m["f1"], B=500, seed=1)
        boots[name] = (ci_pr[2], ci_f1[2])
        out["candidates"][name] = {
            "pr_auc": mean["pr_auc"], "pr_auc_ci95": ci_pr[1], "roc_auc": mean["roc_auc"], "f1": mean["f1"], "f1_ci95": ci_f1[1],
            "precision": mean["precision"], "recall": mean["recall"], "fpr": mean["fpr"],
            "selection_half_pr_auc": float(np.mean(d["es_pr"])) if d["es_pr"] else None,
            "configs_picked": sorted(set(d["cfg"]))[:6]}
    for name in store:
        if name == BASE:
            continue
        a, b = boots[name], boots[BASE]
        m = min(len(a[0]), len(b[0]))
        out["candidates"][name]["vs_baseline"] = {
            "pr_auc_diff": float((a[0][:m] - b[0][:m]).mean()), "pr_auc_diff_ci95": np.percentile(a[0][:m] - b[0][:m], [2.5, 97.5]).tolist(),
            "f1_diff": float((a[1][:m] - b[1][:m]).mean()), "f1_diff_ci95": np.percentile(a[1][:m] - b[1][:m], [2.5, 97.5]).tolist()}
    Path("models/improve_baseline.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t_all:.0f}s")
    order = sorted(out["candidates"], key=lambda k: -out["candidates"][k]["pr_auc"])
    print(f"{'candidate':16s} PR-AUC [95% CI]        F1    prec  recall FPR   | sel-half PR | vs baseline PR-AUC [CI]")
    for k in order:
        c = out["candidates"][k]
        vb = c.get("vs_baseline")
        print(f"{k:16s} {c['pr_auc']:.3f} [{c['pr_auc_ci95'][0]:.2f},{c['pr_auc_ci95'][1]:.2f}]  {c['f1']:.2f}  {c['precision']:.2f}  {c['recall']:.2f}   {c['fpr']*100:.1f}% | "
              f"{(c['selection_half_pr_auc'] or float('nan')):.3f}        | " + (f"{vb['pr_auc_diff']:+.3f} [{vb['pr_auc_diff_ci95'][0]:+.2f},{vb['pr_auc_diff_ci95'][1]:+.2f}]" if vb else "-"))


if __name__ == "__main__":
    main()

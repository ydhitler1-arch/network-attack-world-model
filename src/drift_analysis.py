"""
Why does the threshold drift? (see docs/THRESHOLD_DRIFT.md, docs/DRIFT_CAUSES.md)

Diagnoses the day-to-day shift in BENIGN traffic that makes a threshold calibrated once overshoot on some
days (Thursday, Tuesday) for the neural models:

  1. per-feature shift of each day's benign windows against the other days (standardised mean
     difference, KS statistic), and how big that is compared with the attack signal itself
  2. adversarial validation: can a classifier tell a day's benign windows from the other days'? (AUC with
     blocked cross-validation, so adjacent windows never sit in both train and test); which features give
     the day away
  3. false-alarm drivers: on Thursday, which features separate the benign windows the models flagged
     (false alarms) from the benign windows they did not (uses the cached scores from threshold_drift.py)

    python src/drift_analysis.py        # ~1 min -> models/drift_analysis.json
(run src/threshold_drift.py first so models/bench_drift_scores.pkl exists for step 3)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import threshold_drift as TD  # noqa: E402
from state import STATE_COLUMNS, transform_raw  # noqa: E402

DAYS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def smd(a, b):
    """Standardised mean difference (Cohen's d with pooled sd), per column."""
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
    return (a.mean(0) - b.mean(0)) / np.where(sd < 1e-9, 1, sd)


def echo_analysis(state, folds):
    """Are 'false alarms' benign windows whose 6-window lookback still contains an attack (an echo)?"""
    seg = state["segment"].to_numpy()
    atk = state["window_is_attack"].to_numpy() == 1
    L = 6
    # windows since the most recent attack inside the same day (inf if none)
    since = np.full(len(state), np.inf)
    last = -10**9
    for i in range(len(state)):
        if i == 0 or seg[i] != seg[i - 1]:
            last = -10**9
        since[i] = i - last if last > -10**8 else np.inf
        if atk[i]:
            last = i
    res = {}
    for model in ("logreg", "lstm_clf", "world_model"):
        n_seed = len(folds[0]["scores"][model][0])
        rows = {}
        for si in range(n_seed):
            t_all, al_all, y_all, seg_all = [], [], [], []
            for f in folds:
                sv, st = f["scores"][model][0][si], f["scores"][model][1][si]
                al_all.append(TD.fixed_alarms(sv=sv, yv=f["y_val"], st=st))
                t_all.append(f["t_test"]); y_all.append(f["y_test"]); seg_all.append(f["seg_test"])
            t, al, y, sg = (np.concatenate(x) for x in (t_all, al_all, y_all, seg_all))
            ben = y == 0
            echo = ben & (since[t] <= L - 1)          # an attack inside the lookback (windows t-5 .. t-1)
            clean = ben & ~echo
            entry = {
                "fpr_all_benign": float(al[ben].mean()), "fpr_clean": float(al[clean].mean()), "fpr_echo": float(al[echo].mean()) if echo.any() else None,
                "n_clean": int(clean.sum()), "n_echo": int(echo.sum()),
                "share_of_false_alarms_that_are_echo": float(al[echo].sum() / max(al[ben].sum(), 1)),
                "by_day": {DAYS[d]: {"fpr_clean": float(al[clean & (sg == d)].mean()) if (clean & (sg == d)).any() else None,
                                      "fpr_echo": float(al[echo & (sg == d)].mean()) if (echo & (sg == d)).any() else None,
                                      "n_echo": int((echo & (sg == d)).sum())} for d in DAYS}}
            # alarm rate by windows since the last attack
            buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**6)]
            entry["fpr_by_windows_since_attack"] = {f"{a}-{b if b < 10**6 else '+'}": (float(al[ben & (since[t] >= a) & (since[t] <= b)].mean()) if (ben & (since[t] >= a) & (since[t] <= b)).any() else None) for a, b in buckets}
            entry["fpr_never_attacked_today"] = float(al[ben & np.isinf(since[t])].mean()) if (ben & np.isinf(since[t])).any() else None
            for k, v in entry.items():
                rows.setdefault(k, []).append(v)
        res[model] = {k: ([v for v in vs] if k in ("by_day", "fpr_by_windows_since_attack") else vs) for k, vs in rows.items()}
        # average over seeds
        avg = {}
        for k, vs in rows.items():
            if k in ("by_day", "fpr_by_windows_since_attack"):
                keys = vs[0].keys()
                avg[k] = {kk: (np.mean([x[kk] if not isinstance(x[kk], dict) else 0 for x in vs]) if not isinstance(vs[0][kk], dict) else
                               {k3: (float(np.mean([x[kk][k3] for x in vs if x[kk][k3] is not None])) if any(x[kk][k3] is not None for x in vs) else None) for k3 in vs[0][kk]}) for kk in keys}
            else:
                vals = [v for v in vs if v is not None]
                avg[k] = float(np.mean(vals)) if vals else None
        res[model] = avg
    return res


def val_test_shift(folds):
    """Per fold: how far are the benign TEST scores from the benign VALIDATION scores the threshold was set on?"""
    from scipy.stats import ks_2samp
    res = {}
    for model in ("logreg", "lstm_clf", "world_model"):
        rows = []
        for k, f in enumerate(folds):
            n_seed = len(f["scores"][model][0])
            for si in range(n_seed):
                sv, st = f["scores"][model][0][si], f["scores"][model][1][si]
                bv, bt = sv[f["y_val"] == 0], st[f["y_test"] == 0]
                iqr = np.subtract(*np.percentile(bv, [75, 25])) + 1e-9
                th = np.quantile(bv, 0.98)
                rows.append({"fold": k, "median_shift_iqr": float((np.median(bt) - np.median(bv)) / iqr),
                             "p90_shift_iqr": float((np.quantile(bt, 0.9) - np.quantile(bv, 0.9)) / iqr),
                             "ks": float(ks_2samp(bv, bt).statistic), "n_val_benign": int(len(bv)), "n_above_thr_val": int((bv > th).sum()),
                             "realised_fpr": float((bt > th).mean())})
        df = pd.DataFrame(rows).groupby("fold").mean()
        res[model] = {"by_fold": df.round(3).to_dict(orient="index"),
                      "mean_median_shift_iqr": float(df["median_shift_iqr"].mean()), "mean_p90_shift_iqr": float(df["p90_shift_iqr"].mean()),
                      "mean_ks": float(df["ks"].mean()), "folds_with_fpr_above_4pct": int((df["realised_fpr"] > 0.04).sum()),
                      "fpr_range_over_folds": [float(df["realised_fpr"].min()), float(df["realised_fpr"].max())]}
    return res


def main():
    state = pd.read_parquet("data/state_v1.parquet").reset_index(drop=True)
    seg = state["segment"].to_numpy()
    atk = state["window_is_attack"].to_numpy() == 1
    ben = ~atk
    raw = transform_raw(state, STATE_COLUMNS)
    mu, sd = raw[ben].mean(0), raw[ben].std(0)
    sd = np.where(sd < 1e-9, 1, sd)
    Z = (raw - mu) / sd                    # log1p + z-score on all benign windows (analysis only)
    blocks = np.arange(len(state)) // 60
    out = {"days": {}, "attack_signal": {}}

    # attack signal size, for scale: attack vs benign windows
    a_smd = smd(Z[atk], Z[ben])
    out["attack_signal"] = {"median_abs_smd": float(np.median(np.abs(a_smd))),
                            "top": {STATE_COLUMNS[i]: float(a_smd[i]) for i in np.argsort(-np.abs(a_smd))[:8]}}

    for d, name in DAYS.items():
        m_day, m_rest = ben & (seg == d), ben & (seg != d)
        s = smd(Z[m_day], Z[m_rest])
        ks = np.array([ks_2samp(Z[m_day][:, j], Z[m_rest][:, j]).statistic for j in range(Z.shape[1])])
        order = np.argsort(-np.abs(s))
        # adversarial validation: day vs rest on benign windows only, blocked CV
        Xb, yb, gb = Z[ben], (seg[ben] == d).astype(int), blocks[ben]
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(Xb, yb, gb):
            if yb[te].min() == yb[te].max():
                continue
            clf = LogisticRegression(C=0.3, max_iter=2000).fit(Xb[tr], yb[tr])
            aucs.append(roc_auc_score(yb[te], clf.decision_function(Xb[te])))
        coef = LogisticRegression(C=0.3, max_iter=2000).fit(Xb, yb).coef_[0]
        out["days"][name] = {
            "n_benign": int(m_day.sum()),
            "median_abs_smd": float(np.median(np.abs(s))), "n_features_abs_smd_gt_1": int((np.abs(s) > 1).sum()),
            "top_shifted": {STATE_COLUMNS[i]: {"smd": float(s[i]), "ks": float(ks[i])} for i in order[:8]},
            "adversarial_auc": float(np.mean(aucs)) if aucs else None,
            "adversarial_top_coef": {STATE_COLUMNS[i]: float(coef[i]) for i in np.argsort(-np.abs(coef))[:6]},
        }

    # false-alarm drivers on Thursday and Tuesday (cached scores from threshold_drift.py)
    cache = Path("models/bench_drift_scores.pkl")
    if cache.exists():
        folds = pickle.load(open(cache, "rb"))
        out["false_alarm_drivers"] = {}
        for day in (3, 1):
            for model in ("lstm_clf", "world_model"):
                fa_idx, ok_idx = [], []
                for f in folds:
                    sv, st = f["scores"][model][0][0], f["scores"][model][1][0]
                    al = TD.fixed_alarms(sv=sv, yv=f["y_val"], st=st)
                    m = (f["seg_test"] == day) & (f["y_test"] == 0)
                    fa_idx += list(f["t_test"][m & al])
                    ok_idx += list(f["t_test"][m & ~al])
                fa_idx, ok_idx = np.array(fa_idx), np.array(ok_idx)
                if len(fa_idx) < 5:
                    continue
                s = smd(Z[fa_idx], Z[ok_idx])
                day_shift = smd(Z[ben & (seg == day)], Z[ben & (seg != day)])
                top = np.argsort(-np.abs(s))[:8]
                out["false_alarm_drivers"][f"{DAYS[day]}:{model}"] = {
                    "false_alarm_windows": int(len(fa_idx)), "quiet_windows": int(len(ok_idx)),
                    "top": {STATE_COLUMNS[i]: {"fa_vs_quiet_smd": float(s[i]), "day_shift_smd": float(day_shift[i])} for i in top},
                    "overlap_with_day_shift_top8": sorted(set(STATE_COLUMNS[i] for i in top) &
                                                          set(STATE_COLUMNS[i] for i in np.argsort(-np.abs(day_shift))[:8]))}
    if cache.exists():
        out["echo_analysis"] = echo_analysis(state, folds)
        out["val_test_shift"] = val_test_shift(folds)
    Path("models/drift_analysis.json").write_text(json.dumps(out, indent=1))

    print(f"attack signal: median |SMD| {out['attack_signal']['median_abs_smd']:.2f}; top {list(out['attack_signal']['top'].items())[:3]}")
    for name, d in out["days"].items():
        print(f"\n{name}: {d['n_benign']} benign windows | median |SMD| vs other days {d['median_abs_smd']:.2f} | "
              f"{d['n_features_abs_smd_gt_1']} features with |SMD|>1 | adversarial AUC {d['adversarial_auc']}")
        print("   top shifted:", ", ".join(f"{k} ({v['smd']:+.1f})" for k, v in list(d["top_shifted"].items())[:6]))
    for m, e in out.get("echo_analysis", {}).items():
        print(f"\necho analysis {m}: FPR all benign {e['fpr_all_benign']*100:.1f}% | clean (no attack in lookback) {e['fpr_clean']*100:.1f}% (n={e['n_clean']:.0f}) | "
              f"echo (attack in lookback) {e['fpr_echo']*100 if e['fpr_echo'] is not None else float('nan'):.1f}% (n={e['n_echo']:.0f}) | echo share of false alarms {e['share_of_false_alarms_that_are_echo']*100:.0f}%")
        print("   by windows since last attack:", {k: (round(v*100,1) if v is not None else None) for k, v in e["fpr_by_windows_since_attack"].items()}, "| never attacked today:", round(e["fpr_never_attacked_today"]*100,1) if e["fpr_never_attacked_today"] is not None else None)
        print("   clean-window FPR by day:", {d: (round(x['fpr_clean']*100,1) if x['fpr_clean'] is not None else None) for d, x in e["by_day"].items()})
    for m, v in out.get("val_test_shift", {}).items():
        print(f"\nval->test shift {m}: median shift {v['mean_median_shift_iqr']:+.2f} IQR, p90 shift {v['mean_p90_shift_iqr']:+.2f} IQR, KS {v['mean_ks']:.2f} | realised FPR by fold "
              + " ".join(f"{x['realised_fpr']*100:.1f}" for x in v["by_fold"].values()) + f" | val benign per fold ~{np.mean([x['n_val_benign'] for x in v['by_fold'].values()]):.0f}, windows above the threshold ~{np.mean([x['n_above_thr_val'] for x in v['by_fold'].values()]):.1f}")
    for k, v in out.get("false_alarm_drivers", {}).items():
        print(f"\nfalse alarms {k}: {v['false_alarm_windows']} FA vs {v['quiet_windows']} quiet windows")
        print("   top:", ", ".join(f"{n} ({x['fa_vs_quiet_smd']:+.1f}, day shift {x['day_shift_smd']:+.1f})" for n, x in list(v["top"].items())[:6]))
        print("   overlap with day-shift top-8:", v["overlap_with_day_shift_top8"])


if __name__ == "__main__":
    main()

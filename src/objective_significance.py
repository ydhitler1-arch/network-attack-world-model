"""
Is the world-model objective fix (rollout_task_loss=false) a significant improvement?

Runs the blocked-CV benchmark for the world model twice, original objective (infiltration/stage losses
also on imagined future steps) vs revised (real history steps only), with 5 seeds each, then compares
them with a paired block bootstrap (same resampled blocks for both, as in src/benchmark.py) on PR-AUC
and F1 for the current-window task (k=0) and the 3-minute-ahead task (k=3).

    python src/objective_significance.py        # about 5 min -> models/objective_significance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as B  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 1000


def main():
    base = json.loads(Path(B.CFG_PATH).read_text(encoding="utf-8"))
    assert base.get("rollout_task_loss") is False, "config should hold the revised objective"
    state = pd.read_parquet(base["state_path"]).reset_index(drop=True)
    runs = {}
    for name, flag in (("revised", False), ("original", True)):
        print(f"running world model, {name} objective, seeds {SEEDS}", flush=True)
        cfg = dict(base, rollout_task_loss=flag)
        runs[name] = B.run_cv(cfg, state, tasks=(0, 3), seeds=SEEDS, models=["world_model"], log=lambda *_: None)

    out = {"seeds": list(SEEDS), "bootstrap_replicates": N_BOOT, "tasks": {}}
    for t in (0, 3):
        meta = runs["revised"]["meta"]
        assert np.array_equal(meta["y"][t], runs["original"]["meta"]["y"][t])  # identical test samples
        y, blocks = meta["y"][t], meta["block"]
        rows = np.ones(len(y), bool)
        res = {n: (r["res"]["world_model"][t]["score"], r["res"]["world_model"][t]["thr"]) for n, r in runs.items()}
        entry = {"n_test": int(len(y)), "attack_rate": float(y.mean())}
        for metric in ("pr_auc", "f1"):
            fn = lambda m, k=metric: m[k]
            boots = {n: B.block_bootstrap(y, S, T, blocks, rows, fn, B=N_BOOT, seed=1) for n, (S, T) in res.items()}
            n_rep = min(len(boots["revised"][2]), len(boots["original"][2]))
            diff = boots["revised"][2][:n_rep] - boots["original"][2][:n_rep]
            per_seed = {n: [float(B.point_metrics(y, S[i], T[i])[metric]) for i in range(S.shape[0])] for n, (S, T) in res.items()}
            entry[metric] = {
                "revised": {"mean": float(np.mean(per_seed["revised"])), "ci95": boots["revised"][1], "per_seed": per_seed["revised"]},
                "original": {"mean": float(np.mean(per_seed["original"])), "ci95": boots["original"][1], "per_seed": per_seed["original"]},
                "paired_diff_revised_minus_original": {
                    "mean": float(diff.mean()), "ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                    "share_of_bootstrap_replicates_revised_better": float((diff > 0).mean())},
            }
        out["tasks"][f"k{t}"] = entry
    Path("models/objective_significance.json").write_text(json.dumps(out, indent=1))
    for k, e in out["tasks"].items():
        for m in ("pr_auc", "f1"):
            d = e[m]["paired_diff_revised_minus_original"]
            print(f"{k} {m}: original {e[m]['original']['mean']:.3f}  revised {e[m]['revised']['mean']:.3f}  "
                  f"diff {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  revised better in {d['share_of_bootstrap_replicates_revised_better']:.0%} of resamples")
    print("saved models/objective_significance.json")


if __name__ == "__main__":
    main()
